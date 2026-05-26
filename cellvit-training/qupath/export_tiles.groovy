/**
 * export_tiles.groovy
 * -------------------
 * Export 1024×1024 px PNG tiles + per-tile cell label CSVs from an H&E image
 * for CellViT++ training.
 *
 * Cell positions and labels come from QuPath DETECTION OBJECTS in the open
 * image (post StarDist + QuST XeniumAnnotation + load_mapping.groovy).
 * The class-name string on each detection's PathClass is mapped to a class_int
 * via label_map.yaml (hand-authored canonical int ↔ label-name table).
 *
 * Auto-routing (default): the tissue is derived from the image URI's
 * `data/xenium/<tissue>/...` segment, and OUTPUT_ROOT is set to
 *     <repo>/cellvit-training/trainingset/<tissue>
 * so a single QuPath CLI batch over the global project exports all tissues
 * in one pass (resumable per PNG).
 *
 * Override (optional): set FORCE_TISSUE below (or pass via QuPath
 * `script -a <tissue>`) to (i) force OUTPUT_ROOT for the active image and
 * (ii) skip images whose URI does not match `data/xenium/<FORCE_TISSUE>/`.
 *
 * Overlap (optional): pass a second `-a <overlap>` (float in [0.0, 1.0))
 * to set OVERLAP_RATIO at the CLI, e.g.
 *     QuPath script -p ... -a heart -a 0.5 export_tiles.groovy
 * Use 0.5 for single-slide tissues (heart, brain, cervix, prostate,
 * lymph_node) to densify the training set; keep 0.0 for multi-slide tissues.
 *
 * Requirements:
 *   - Open the image in QuPath; its .qpdata must already contain detections
 *     whose PathClass names match keys in label_map.yaml (e.g. "tumor",
 *     "lymphoid", …).
 *   - label_map.yaml lives at ${OUTPUT_ROOT}/label_map.yaml.
 *   - SAMPLE_TAG is derived automatically from the image filename.
 *
 * The script is resumable: tiles whose PNG already exists are skipped.
 */

import java.awt.Color
import java.awt.image.BufferedImage
import java.nio.file.Paths
import javax.imageio.ImageIO

// ======================== USER CONFIGURATION ========================
// Leave FORCE_TISSUE = null to auto-route per image (recommended for a
// single CLI batch over the global project). Set to e.g. "breast" to
// restrict export to one tissue. Also picked up from `args` if provided.
def FORCE_TISSUE = null
def SPLIT        = "train"  // "train" or "test"
if (args != null && args.size() > 0 && args[0]) FORCE_TISSUE = args[0].toString()

// Overlap fraction: 0.0 = no overlap (stride = TILE_PX)
//                   0.5 = 50% overlap (stride = 512 px for 1024 tile)
// Also picked up from `args[1]` if provided, e.g.
//     QuPath script -p ... -a heart -a 0.5 export_tiles.groovy
// Recommended for single-slide tissues (heart, brain, cervix, prostate,
// lymph_node) to expand the training set; multi-slide tissues should keep 0.0.
double OVERLAP_RATIO = 0.0
if (args != null && args.size() > 1 && args[1]) {
    try { OVERLAP_RATIO = args[1].toString().toDouble() }
    catch (NumberFormatException e) {
        println "ERROR: args[1]='${args[1]}' is not a valid overlap ratio float."
        return
    }
    if (OVERLAP_RATIO < 0.0 || OVERLAP_RATIO >= 1.0) {
        println "ERROR: OVERLAP_RATIO must be in [0.0, 1.0); got ${OVERLAP_RATIO}."
        return
    }
}

// Augmentations applied to each base tile (coordinate-aware)
// Each enabled augmentation writes an additional PNG+CSV pair.
// NOTE: prefer leaving these off and using CellViT++'s YAML
// `transformations:` block (RandomRotate90 + HFlip + VFlip), which applies
// random augs at every epoch without disk overhead.
boolean AUG_HFLIP  = false   // horizontal flip
boolean AUG_VFLIP  = false   // vertical flip
boolean AUG_ROT90  = false   // rotate 90° CCW
boolean AUG_ROT180 = false   // rotate 180°
boolean AUG_ROT270 = false   // rotate 270° CCW
// ====================================================================

// ── Export parameters (fixed per spec)
final double EXPORT_MPP = 0.25          // µm/px target resolution
final int    TILE_PX    = 1024          // tile size in output pixels
final int    MIN_CELLS  = 5             // skip tiles with fewer assigned cells
final double BG_THRESH  = 240.0         // mean RGB > this → background, skip

// Derived from OVERLAP_RATIO
int    STRIDE_PX = (int) Math.round(TILE_PX * (1.0 - OVERLAP_RATIO))
double STRIDE_UM = STRIDE_PX * EXPORT_MPP
double TILE_UM   = TILE_PX   * EXPORT_MPP

// ── Validate image is open
def imageData = getCurrentImageData()
if (imageData == null) { println "ERROR: No image open in QuPath."; return }

def server   = imageData.getServer()
def slideMPP = server.getPixelCalibration().getPixelWidthMicrons()
if (Double.isNaN(slideMPP) || slideMPP <= 0) {
    println "ERROR: No pixel calibration set on this image. Set it in Image > Image Info."
    return
}

// ── Derive SAMPLE_TAG from the open image filename
//    e.g. "Xenium_Prime_Breast_Cancer_FFPE_he_image.ome.tif" → "Xenium_Prime_Breast_Cancer_FFPE_he_image"
def imageName = server.getMetadata().getName()
def SAMPLE_TAG = imageName
    .replaceAll(/\s*-\s*Image\d+$/, "")   // strip Bio-Formats series suffix e.g. " - Image0"
    .replaceAll(/(?i)\.ome\.tif$/, "")
    .replaceAll(/(?i)\.tif$/, "")

// ── Derive tissue + OUTPUT_ROOT from image URI segment "data/xenium/<tissue>/..."
//    Then OUTPUT_ROOT = <repo>/cellvit-training/trainingset/<tissue>.
//    Use server.getURIs() (a clean Collection<URI>) instead of server.getPath()
//    because the latter returns the QuPath canonical string
//    "BioFormatsImageServer: file:/.../foo.ome.tif [--series, 0]" which the
//    File constructor below misparses.
def uriList = server.getURIs() as List
if (uriList.isEmpty()) {
    println "ERROR: server.getURIs() is empty; cannot derive tissue."
    return
}
def serverURI = uriList[0]
def serverPath = serverURI.getPath()  // "/workspace/.../data/xenium/<tissue>/..."
def tissueMatch = (serverPath =~ /data\/xenium\/([^\/]+)\//)
if (!tissueMatch.find()) {
    println "WARN: image URI does not contain 'data/xenium/<tissue>/' segment; skipping."
    println "      URI = ${serverURI}"
    return
}
def TISSUE = tissueMatch.group(1)
if (FORCE_TISSUE != null && TISSUE != FORCE_TISSUE) {
    println "Skip: image tissue '${TISSUE}' != FORCE_TISSUE '${FORCE_TISSUE}'"
    return
}

// Resolve repo root from the same URI: keep everything before "data/xenium/".
def repoRootStr = serverPath.substring(0, tissueMatch.start()).replaceFirst(/^file:\/+/, "/")
// Strip URL artefacts the JVM may leave on Windows ("file:/C:/...").
if (repoRootStr.matches(/^\/[A-Za-z]:.*/)) repoRootStr = repoRootStr.substring(1)
def repoRoot = new File(repoRootStr).getCanonicalFile()
def OUTPUT_ROOT = new File(repoRoot, "cellvit-training/trainingset/${TISSUE}").getAbsolutePath()

// ── Load label_map.yaml  (int -> label-name)  →  invert to (label-name -> int)
def LABEL_MAP_YAML = "${OUTPUT_ROOT}/label_map.yaml"
def labelMapFile = new File(LABEL_MAP_YAML)
if (!labelMapFile.exists()) {
    println "ERROR: label_map.yaml not found: ${LABEL_MAP_YAML}"
    println "Run/edit it by hand; it is the canonical int ↔ label-name table."
    return
}
def labelToInt = [:]
labelMapFile.eachLine { line ->
    def m = (line =~ /^\s*(\d+)\s*:\s*"?([^"#]+?)"?\s*(#.*)?$/)
    if (m.find()) {
        int ci = m.group(1).toInteger()
        String nm = m.group(2).trim()
        labelToInt[nm] = ci
    }
}
if (labelToInt.isEmpty()) {
    println "ERROR: no entries parsed from ${LABEL_MAP_YAML}"
    return
}
println "Loaded ${labelToInt.size()} classes from label_map.yaml: ${labelToInt}"

double ds = EXPORT_MPP / slideMPP    // QuPath downsample factor

// ── Build augmentation list: each entry is [suffix, transform closure]
//    Transform closures take (img, cellList, tilePX) and return [newImg, newCells]
def augmentations = []
if (AUG_HFLIP)  augmentations << ["_hflip",  { img, cs, sz -> hflip(img, cs, sz)  }]
if (AUG_VFLIP)  augmentations << ["_vflip",  { img, cs, sz -> vflip(img, cs, sz)  }]
if (AUG_ROT90)  augmentations << ["_rot90",  { img, cs, sz -> rot90(img, cs, sz)  }]
if (AUG_ROT180) augmentations << ["_rot180", { img, cs, sz -> rot180(img, cs, sz) }]
if (AUG_ROT270) augmentations << ["_rot270", { img, cs, sz -> rot270(img, cs, sz) }]

println "=== export_tiles.groovy ==="
println "Image      : ${imageName}"
println "Tissue     : ${TISSUE}${FORCE_TISSUE ? '  (forced)' : '  (auto-routed)'}"
println "Output root: ${OUTPUT_ROOT}"
println "Sample tag : ${SAMPLE_TAG}"
println "Label map  : ${LABEL_MAP_YAML}"
println "Slide size : ${server.getWidth()} × ${server.getHeight()} px"
println "Slide MPP  : ${slideMPP} µm/px   Downsample: ${String.format('%.4f', ds)}"
println "Split      : ${SPLIT}"
println "Overlap    : ${(OVERLAP_RATIO*100).intValue()}%  (tile ${TILE_PX}px, stride ${STRIDE_PX}px)"
println "Augment    : ${augmentations.collect{it[0]}.join(', ') ?: 'none'}"

// ── Create output directories
def imgDir = new File("${OUTPUT_ROOT}/${SPLIT}/images")
def lblDir = new File("${OUTPUT_ROOT}/${SPLIT}/labels")
imgDir.mkdirs()
lblDir.mkdirs()

// ── Load cells from QuPath DETECTION OBJECTS in the open image.
//    Centroid is in full-res image pixels; convert to µm so the existing
//    bucketing math (which works in µm) is reused unchanged.
println "\nLoading cells from detection objects ..."
def detections = getDetectionObjects()
println "  ${detections.size()} detection objects in current image"

def cells = []
int nSkipNoClass = 0
int nSkipUnmapped = 0
def unmappedClasses = [:].withDefault { 0 }
detections.each { obj ->
    def pc = obj.getPathClass()
    if (pc == null) { nSkipNoClass++; return }
    def ci = labelToInt.get(pc.getName())
    if (ci == null) {
        nSkipUnmapped++
        unmappedClasses[pc.getName()]++
        return
    }
    def roi = obj.getROI()
    cells << [x: roi.getCentroidX() * slideMPP,
              y: roi.getCentroidY() * slideMPP,
              cls: ci]
}
println "  ${cells.size()} cells kept  (${nSkipNoClass} no PathClass, ${nSkipUnmapped} unmapped class)"
if (!unmappedClasses.isEmpty()) {
    println "  Unmapped PathClass names (not in label_map.yaml):"
    unmappedClasses.each { k, v -> println "    ${k}: ${v}" }
}
if (cells.isEmpty()) {
    println "ERROR: zero classifiable detections. Did StarDist + load_mapping.groovy run?"
    return
}

// ── Pre-bucket cells by tile grid position
//    A cell at (x_um, y_um) belongs to tile (col, row) when:
//      col*STRIDE_UM <= x_um < col*STRIDE_UM + TILE_UM
//      row*STRIDE_UM <= y_um < row*STRIDE_UM + TILE_UM
//
//    Because TILE_UM > STRIDE_UM (overlap = 16 µm), a cell may belong to
//    up to 4 tiles.  We bucket each cell into every tile it belongs to so
//    that the export loop never needs to scan all cells.
println "Bucketing cells into tile grid ..."

def buckets = new HashMap<Long, List>()
cells.each { c ->
    int c0 = (int) Math.max(0, (long) Math.floor((c.x - TILE_UM) / STRIDE_UM) + 1)
    int c1 = (int) Math.floor(c.x / STRIDE_UM)
    int r0 = (int) Math.max(0, (long) Math.floor((c.y - TILE_UM) / STRIDE_UM) + 1)
    int r1 = (int) Math.floor(c.y / STRIDE_UM)
    for (int col = c0; col <= c1; col++) {
        for (int row = r0; row <= r1; row++) {
            long key = ((long) col << 32) | (row & 0xFFFFFFFFL)
            if (!buckets.containsKey(key)) buckets[key] = []
            buckets[key].add(c)
        }
    }
}
println "  ${buckets.size()} non-empty tile positions"

// ── Tile grid extents
double slideWidthUM  = (double) server.getWidth()  * slideMPP
double slideHeightUM = (double) server.getHeight() * slideMPP
int nCols = (int) Math.ceil(slideWidthUM  / STRIDE_UM)
int nRows = (int) Math.ceil(slideHeightUM / STRIDE_UM)
println "\nGrid: ${nCols} cols × ${nRows} rows (${nCols * nRows} positions)"

// ── Emit per-slide tile-geometry sidecar (used by audit_split_reuse.py)
//    Captures every constant needed to recover any tile's slide-coord bbox
//    from its filename: tileIdx → (col,row) = ((tileIdx-1) // nCols,
//    (tileIdx-1) % nCols), origin_µm = (col*STRIDE_UM, row*STRIDE_UM).
def geomDir = new File("${OUTPUT_ROOT}/${SPLIT}/tile_geometry")
geomDir.mkdirs()
def geomFile = new File(geomDir, "${SAMPLE_TAG}.json")
geomFile.withWriter { wrt ->
    wrt.write("{\n")
    wrt.write("  \"sample_tag\": \"${SAMPLE_TAG}\",\n")
    wrt.write("  \"image_name\": \"${imageName}\",\n")
    wrt.write("  \"tissue\": \"${TISSUE}\",\n")
    wrt.write("  \"slide_width_px\": ${server.getWidth()},\n")
    wrt.write("  \"slide_height_px\": ${server.getHeight()},\n")
    wrt.write("  \"slide_mpp\": ${slideMPP},\n")
    wrt.write("  \"export_mpp\": ${EXPORT_MPP},\n")
    wrt.write("  \"tile_px\": ${TILE_PX},\n")
    wrt.write("  \"stride_px\": ${STRIDE_PX},\n")
    wrt.write("  \"overlap_ratio\": ${OVERLAP_RATIO},\n")
    wrt.write("  \"n_cols\": ${nCols},\n")
    wrt.write("  \"n_rows\": ${nRows}\n")
    wrt.write("}\n")
}
println "Geometry  : ${geomFile.getAbsolutePath()}"

println "Starting tile export ...\n"

int tileIdx    = 0   // sequential index over ALL grid positions (used for naming)
int nWritten   = 0
int nResumed   = 0
int nSkipCells = 0
int nSkipBg    = 0
long startTime = System.currentTimeMillis()

for (int row = 0; row < nRows; row++) {
    for (int col = 0; col < nCols; col++) {
        tileIdx++

        // ── 1. Cell filter (cheap — avoids image I/O)
        long key   = ((long) col << 32) | (row & 0xFFFFFFFFL)
        def tc     = buckets.get(key)
        if (tc == null || tc.size() < MIN_CELLS) { nSkipCells++; continue }

        // ── 2. Resume check
        def stem    = "${SAMPLE_TAG}_tile_${String.format('%05d', tileIdx)}"
        def imgFile = new File(imgDir, "${stem}.png")
        if (imgFile.exists()) { nResumed++; continue }

        // ── 3. Compute region in full-res pixels
        double ox = col * STRIDE_UM
        double oy = row * STRIDE_UM
        int rx    = (int) Math.round(ox / slideMPP)
        int ry    = (int) Math.round(oy / slideMPP)
        int rw    = (int) Math.round(TILE_UM / slideMPP)
        int rh    = (int) Math.round(TILE_UM / slideMPP)

        // Clamp to slide bounds (edge tiles)
        if (rx >= server.getWidth() || ry >= server.getHeight()) { nSkipCells++; continue }
        int crw = Math.min(rw, server.getWidth()  - rx)
        int crh = Math.min(rh, server.getHeight() - ry)

        def request = RegionRequest.createInstance(server.getPath(), ds, rx, ry, crw, crh, 0, 0)

        // ── 4. Read image region
        def rawImg = server.readRegion(request)
        if (rawImg == null) { nSkipCells++; continue }

        // ── 5. Build exactly TILE_PX × TILE_PX RGB image (white-pads edge tiles)
        def rgbImg = new BufferedImage(TILE_PX, TILE_PX, BufferedImage.TYPE_INT_RGB)
        def g2d    = rgbImg.createGraphics()
        g2d.setColor(Color.WHITE)
        g2d.fillRect(0, 0, TILE_PX, TILE_PX)
        g2d.drawImage(rawImg, 0, 0, null)
        g2d.dispose()

        // ── 6. Tissue mask: sample every 8th pixel, skip if mean brightness > threshold
        long bsum  = 0L
        int  step  = 8
        int  nsamp = 0
        for (int py = 0; py < TILE_PX; py += step) {
            for (int px = 0; px < TILE_PX; px += step) {
                int pix = rgbImg.getRGB(px, py)
                bsum += ((pix >> 16) & 0xFF) + ((pix >> 8) & 0xFF) + (pix & 0xFF)
                nsamp += 3
            }
        }
        if ((double) bsum / nsamp > BG_THRESH) { nSkipBg++; continue }

        // ── 7. Collect cell pixel coords for this tile
        def tileCells = []
        tc.each { c ->
            int cpx = (int) Math.round((c.x - ox) / EXPORT_MPP)
            int cpy = (int) Math.round((c.y - oy) / EXPORT_MPP)
            if (cpx >= 0 && cpx < TILE_PX && cpy >= 0 && cpy < TILE_PX)
                tileCells << [x: cpx, y: cpy, cls: c.cls]
        }

        // ── 8. Write base PNG + CSV
        ImageWriterTools.writeImage(rgbImg, imgFile.getAbsolutePath())
        def lblFile = new File(lblDir, "${stem}.csv")
        lblFile.withWriter { wrt ->
            tileCells.each { wrt.writeLine("${it.x},${it.y},${it.cls}") }
        }
        nWritten++

        // ── 9. Write augmented variants
        augmentations.each { aug ->
            def augStem    = "${stem}${aug[0]}"
            def augImgFile = new File(imgDir, "${augStem}.png")
            if (augImgFile.exists()) { nResumed++; return }
            def (augImg, augCells) = aug[1](rgbImg, tileCells, TILE_PX)
            ImageWriterTools.writeImage(augImg, augImgFile.getAbsolutePath())
            new File(lblDir, "${augStem}.csv").withWriter { wrt ->
                augCells.each { wrt.writeLine("${it.x},${it.y},${it.cls}") }
            }
            nWritten++
        }
        if (nWritten % 200 == 0) {
            long elapsed = (System.currentTimeMillis() - startTime) / 1000
            println "  [${new java.text.SimpleDateFormat('HH:mm:ss').format(new Date())}] ${nWritten} tiles written" +
                    "  (${nSkipCells} skip-cells, ${nSkipBg} skip-bg, ${nResumed} resumed)" +
                    "  ${elapsed}s elapsed"
        }
    }
}

long totalSec = (System.currentTimeMillis() - startTime) / 1000
println "\n=== Done ==="
println "  Written  : ${nWritten} tiles (base + augmented)"
println "  Resumed  : ${nResumed} (already existed – skipped)"
println "  Skipped  : ${nSkipCells} (< ${MIN_CELLS} cells)  |  ${nSkipBg} (background)"
println "  Time     : ${totalSec}s"
println "  Output   : ${imgDir.getAbsolutePath()}"

// ── Augmentation helper functions
// Each takes (BufferedImage img, List cells, int sz) → [BufferedImage, List]

def hflip(img, cells, sz) {
    def out = new BufferedImage(sz, sz, BufferedImage.TYPE_INT_RGB)
    def g = out.createGraphics()
    g.drawImage(img, sz, 0, -sz, sz, null)
    g.dispose()
    def nc = cells.collect { [x: sz - 1 - it.x, y: it.y, cls: it.cls] }
    [out, nc]
}

def vflip(img, cells, sz) {
    def out = new BufferedImage(sz, sz, BufferedImage.TYPE_INT_RGB)
    def g = out.createGraphics()
    g.drawImage(img, 0, sz, sz, -sz, null)
    g.dispose()
    def nc = cells.collect { [x: it.x, y: sz - 1 - it.y, cls: it.cls] }
    [out, nc]
}

def rot90(img, cells, sz) {
    // 90° CCW: (x,y) → (y, sz-1-x)
    def out = new BufferedImage(sz, sz, BufferedImage.TYPE_INT_RGB)
    def g = out.createGraphics()
    g.translate(0, sz)
    g.rotate(-Math.PI / 2)
    g.drawImage(img, 0, 0, null)
    g.dispose()
    def nc = cells.collect { [x: it.y, y: sz - 1 - it.x, cls: it.cls] }
    [out, nc]
}

def rot180(img, cells, sz) {
    def out = new BufferedImage(sz, sz, BufferedImage.TYPE_INT_RGB)
    def g = out.createGraphics()
    g.drawImage(img, sz, sz, -sz, -sz, null)
    g.dispose()
    def nc = cells.collect { [x: sz - 1 - it.x, y: sz - 1 - it.y, cls: it.cls] }
    [out, nc]
}

def rot270(img, cells, sz) {
    // 270° CCW = 90° CW: (x,y) → (sz-1-y, x)
    def out = new BufferedImage(sz, sz, BufferedImage.TYPE_INT_RGB)
    def g = out.createGraphics()
    g.translate(sz, 0)
    g.rotate(Math.PI / 2)
    g.drawImage(img, 0, 0, null)
    g.dispose()
    def nc = cells.collect { [x: sz - 1 - it.y, y: it.x, cls: it.cls] }
    [out, nc]
}
