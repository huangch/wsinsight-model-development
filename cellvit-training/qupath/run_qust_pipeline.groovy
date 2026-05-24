/**
 * run_qust_pipeline.groovy
 * ------------------------
 * Per-image QuST ritual for the pantissue CellViT training set:
 *
 *   1. Whole-slide rectangle annotation -- universal root parent that
 *                                     does NOT depend on stain intensity
 *                                     or scanner. False-positive nuclei in
 *                                     background are filtered downstream
 *                                     by XeniumAnnotation (drops detections
 *                                     with no matching Xenium cell) and by
 *                                     export_tiles.groovy (MIN_CELLS + BG
 *                                     mean-RGB threshold per tile).
 *   2. StarDistCellNucleusDetection -- detect H&E nuclei inside the rect
 *   3. XeniumAnnotation            -- transfer Xenium cluster_id onto each
 *                                     H&E detection (PathClass.name = cluster_id)
 *
 * Designed for headless CLI batch over every image in the QuPath project:
 *
 *   QuPath script -s -p data/qprj/project.qpproj \
 *       cellvit-training/qupath/run_qust_pipeline.groovy
 *
 * `-s` persists the resulting tissue annotation, nucleus detections and
 * Xenium-derived PathClasses into each image's .qpdata. The script must be
 * followed by `load_mapping.groovy` (cluster_id -> pantissue label) and
 * then `export_tiles.groovy`.
 *
 * Per-image Xenium outs dir is derived from the image URI: the script
 * looks for `<image-parent-dir>/outs`. Every sample in data/xenium/<tissue>/
 * follows this layout, so no per-image config file is needed.
 *
 * QuST settings used implicitly:
 *   - QuST > Set StarDist model location   (must point at a dir containing
 *     the H&E nucleus model .pb file referenced by STARDIST_MODEL below)
 */
import qupath.lib.scripting.QP
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.objects.PathObjects
import qupath.lib.objects.classes.PathClass
import qupath.lib.regions.ImagePlane
import qupath.lib.roi.ROIs
import qupath.lib.gui.QuPathGUI
import qupath.fx.dialogs.Dialogs
import qupath.fx.dialogs.FileChoosers
import java.nio.file.Paths

//------------------------------------------------------------------------------
// Config -- edit before batch run
//------------------------------------------------------------------------------
def STARDIST_MODEL = "he_heavy_augment.pb"   // file name inside QuST's
                                             // configured stardist model dir
def CELL_EXPANSION_UM = 5.0                  // nucleus -> cell expansion (um);
                                             //   set to -1 for nuclei only
def PROB_THRESHOLD   = 0.5                   // StarDist detection probability

// Xenium loader behaviour (matches QuST defaults except removeUnlabeledCells)
def XENIUM_DONT_TRANSFORM    = false
def XENIUM_AFFINE_ONLY       = false
def XENIUM_REMOVE_UNLABELED  = true
def XENIUM_MASK_DOWNSAMPLE   = 2

//------------------------------------------------------------------------------
// Resolve current image + matching Xenium outs/ dir
//------------------------------------------------------------------------------
def imageData = getCurrentImageData()
if (imageData == null) {
    println "[run_qust_pipeline] no current image -- skipping"
    return
}
def server = imageData.getServer()
def imgName = GeneralTools.getNameWithoutExtension(server.getMetadata().getName())
println "==== ${imgName} ===="

// QuPath project entries store the image URI; for a local OME-TIFF, the
// parent directory of that URI is the Xenium sample folder.
def uri = server.getURIs().iterator().next()
def imgPath = Paths.get(uri)
def xeniumDir = imgPath.getParent().resolve("outs").toString()
def outsFile = new File(xeniumDir)
if (!outsFile.isDirectory()) {
    // Auto-detected location missing. If a GUI is available (interactive
    // QuPath session), prompt the user to pick the outs/ folder manually;
    // if running headless (CLI batch), fall back to printing [skip].
    def gui = QuPathGUI.getInstance()
    if (gui != null) {
        Dialogs.showWarningNotification(
                "Xenium outs/ not found",
                "Auto-detected path does not exist:\n${xeniumDir}\nPlease select the Xenium outs/ folder for ${imgName}.")
        def picked = FileChoosers.promptForDirectory(
                "Select Xenium outs/ folder for ${imgName}",
                outsFile.getParentFile())
        if (picked == null) {
            println "  [skip] user cancelled outs/ folder selection"
            return
        }
        outsFile = picked
        xeniumDir = picked.getAbsolutePath()
    } else {
        println "  [skip] Xenium outs/ not found at: ${xeniumDir}"
        return
    }
}
println "  xeniumDir = ${xeniumDir}"

//------------------------------------------------------------------------------
// Clear any prior annotations/detections so the script is idempotent
//------------------------------------------------------------------------------
clearAllObjects()

//------------------------------------------------------------------------------
// 1) Whole-slide rectangle annotation (root parent)
//
// Rationale: a stain/scanner-dependent tissue-detection threshold is fragile
// across cohorts. Instead we annotate the entire slide as one rectangle and
// let StarDist run everywhere. False-positive nuclei in background will be
// removed by XeniumAnnotation (removeUnlabeledCells=true keeps only
// detections that match a real Xenium cell), and empty/background tiles will
// be dropped by export_tiles.groovy (MIN_CELLS + BG_THRESH filters).
//------------------------------------------------------------------------------
println "  [1/3] Whole-slide rectangle annotation ..."
def wsRoi = ROIs.createRectangleROI(
        0.0, 0.0,
        (double) server.getWidth(), (double) server.getHeight(),
        ImagePlane.getDefaultPlane())
def wsAnnotation = PathObjects.createAnnotationObject(wsRoi)
addObject(wsAnnotation)
println "  whole-slide annotation: ${server.getWidth()} x ${server.getHeight()} px"

def annotations = getAnnotationObjects()
if (annotations.isEmpty()) {
    println "  [skip] failed to create whole-slide annotation"
    return
}

//------------------------------------------------------------------------------
// 2) StarDist H&E nucleus detection inside the tissue annotation(s)
//------------------------------------------------------------------------------
println "  [2/3] StarDistCellNucleusDetection (model=${STARDIST_MODEL}) ..."
selectAnnotations()
def stardistParams = String.format(
        '{"threshold":%.3f,' +
        '"normalizePercentilesLow":1.0,' +
        '"normalizePercentilesHigh":99.0,' +
        '"includeProbability":false,' +
        '"measureShape":false,' +
        '"measureIntensity":false,' +
        '"starDistModel":"%s",' +
        '"channel":"",' +
        '"cellExpansion":%.2f,' +
        '"cellConstrainScale":-1.0,' +
        '"nThreads":0,' +
        '"tileSize":0}',
        PROB_THRESHOLD, STARDIST_MODEL, CELL_EXPANSION_UM)
runPlugin("qupath.ext.qust.StarDistCellNucleusDetection", stardistParams)

def nDet = getDetectionObjects().size()
println "  detections: ${nDet}"
if (nDet == 0) {
    println "  [skip] no detections -- XeniumAnnotation has nothing to label"
    return
}

//------------------------------------------------------------------------------
// 3) XeniumAnnotation: assign cluster_id -> PathClass on each detection
//------------------------------------------------------------------------------
println "  [3/3] XeniumAnnotation ..."
def xeniumDirJson = xeniumDir.replace("\\", "\\\\").replace("\"", "\\\"")
def xeniumParams = String.format(
        '{"xeniumDir":"%s",' +
        '"dontTransform":%b,' +
        '"AffineTransformOnly":%b,' +
        '"removeUnlabeledCells":%b,' +
        '"inclGeneExpr":true,' +
        '"inclBlankCodeword":false,' +
        '"inclUnassignedCodeword":false,' +
        '"inclDeprecatedCodeword":false,' +
        '"inclIntergenicRegion":false,' +
        '"inclNegCtrlCodeword":false,' +
        '"inclNegCtrlProbe":false,' +
        '"maskDownsampling":%d}',
        xeniumDirJson,
        XENIUM_DONT_TRANSFORM,
        XENIUM_AFFINE_ONLY,
        XENIUM_REMOVE_UNLABELED,
        XENIUM_MASK_DOWNSAMPLE)
runPlugin("qupath.ext.qust.XeniumAnnotation", xeniumParams)

//------------------------------------------------------------------------------
// Summary
//------------------------------------------------------------------------------
def labeled = getDetectionObjects().findAll { it.getPathClass() != null }
def tally = [:].withDefault { 0 }
labeled.each { tally[it.getPathClass().getName()] += 1 }
println "  labeled detections: ${labeled.size()} / ${getDetectionObjects().size()}"
println "  cluster_id tally (top 10):"
tally.sort { -it.value }.take(10).each { k, v -> println "    ${k}: ${v}" }
println "done."
