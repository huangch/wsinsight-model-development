/**
 * export_tiles.groovy
 * -------------------
 * Export 1024×1024 px PNG tiles + per-tile cell label CSVs from a Xenium
 * H&E image for CellViT++ training.
 *
 * Requirements:
 *   - Open the sample's *_he_image.ome.tif (or *_he_unaligned_image.ome.tif)
 *     in QuPath before running.
 *   - SAMPLE_TAG and LABEL_CSV are derived automatically from the image filename.
 *   - Only set OUTPUT_ROOT and SPLIT below.
 *
 * The script is resumable: tiles whose PNG already exists are skipped.
 */

import java.awt.Color
import javax.imageio.ImageIO

// ======================== USER CONFIGURATION ========================
// Output trainingset for the tissue currently being exported, e.g.:
//   /workspace/wsinsight/model-development/cellvit-training/trainingset/breast
//   /workspace/wsinsight/model-development/cellvit-training/trainingset/colorectal
// Edit before each run, or set with the QuPath Script Editor "args" hook.
def OUTPUT_ROOT  = "/workspace/wsinsight/model-development/cellvit-training/trainingset/colorectal"
def SPLIT        = "train"  // "train" or "test"

// Overlap fraction: 0.0 = no overlap (stride = TILE_PX)
//                   0.5 = 50% overlap (stride = 512 px for 1024 tile)
double OVERLAP_RATIO = 0.0

// Augmentations applied to each base tile (coordinate-aware)
// Each enabled augmentation writes an additional PNG+CSV pair.
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

// ── Derive SAMPLE_TAG and LABEL_CSV from the open image filename
//    e.g. "Xenium_Prime_Breast_Cancer_FFPE_he_image.ome.tif" → "Xenium_Prime_Breast_Cancer_FFPE_he_image"
def imageName = server.getMetadata().getName()
def SAMPLE_TAG = imageName
    .replaceAll(/\s*-\s*Image\d+$/, "")   // strip Bio-Formats series suffix e.g. " - Image0"
    .replaceAll(/(?i)\.ome\.tif$/, "")
    .replaceAll(/(?i)\.tif$/, "")
def LABEL_CSV  = "${OUTPUT_ROOT}/cell_labels_${SAMPLE_TAG}.csv"

if (!new File(LABEL_CSV).exists()) {
    println "ERROR: Label CSV not found: ${LABEL_CSV}"
    println "Expected filename derived from open image: ${imageName}"
    println "Run build_cell_labels.py first, or check that the correct image is open."
    return
}

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
println "Sample tag : ${SAMPLE_TAG}"
println "Label CSV  : ${LABEL_CSV}"
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

// ── Load cell label CSV  (header: cell_id, x_um, y_um, class_int)
println "\nLoading cell labels from:\n  ${LABEL_CSV} ..."
def labelFile = new File(LABEL_CSV)
if (!labelFile.exists()) { println "ERROR: Label CSV not found: ${LABEL_CSV}"; return }

def cells = []
labelFile.withReader { reader ->
    reader.readLine()   // skip header
    String line
    while ((line = reader.readLine()) != null) {
        def p = line.split(",")
        if (p.length < 4) continue
        cells << [x: p[1].toDouble(), y: p[2].toDouble(), cls: p[3].toInteger()]
    }
}
println "  ${cells.size()} cells loaded"

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
