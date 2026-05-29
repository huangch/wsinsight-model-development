/**
 * reset_clusters.groovy
 * ---------------------
 * Re-runs ONLY QuST's XeniumAnnotation on existing detections to reset their
 * PathClass back to the Xenium cluster_id. Use this when load_mapping.groovy
 * was already run with an outdated celltype_assignment_*.csv and you want to
 * apply a fresh CSV without redoing tissue detection + StarDist.
 *
 * Usage (CLI batch over project):
 *   QuPath script -s -p data/qprj/project.qpproj \
 *       cellvit-training/qupath/reset_clusters.groovy
 *
 * Then re-run load_mapping.groovy to apply the new label mapping.
 *
 * Mirrors XeniumAnnotation config from run_qust_pipeline.groovy exactly.
 */
import java.nio.file.Paths

// Xenium loader behaviour (must match run_qust_pipeline.groovy)
def XENIUM_DONT_TRANSFORM    = false
def XENIUM_AFFINE_ONLY       = false
def XENIUM_REMOVE_UNLABELED  = true
def XENIUM_MASK_DOWNSAMPLE   = 2

def imageData = getCurrentImageData()
if (imageData == null) {
    println "[reset_clusters] no current image -- skipping"
    return
}
def server = imageData.getServer()
def imgName = GeneralTools.getNameWithoutExtension(server.getMetadata().getName())
println "==== ${imgName} ===="

def uri = server.getURIs().iterator().next()
def imgPath = Paths.get(uri)
def xeniumDir = imgPath.getParent().resolve("outs").toString()
def outsFile = new File(xeniumDir)
if (!outsFile.isDirectory()) {
    println "  [skip] Xenium outs/ not found at: ${xeniumDir}"
    return
}

def nBefore = getDetectionObjects().size()
println "  detections before: ${nBefore}"
if (nBefore == 0) {
    println "  [skip] no existing detections -- need to run run_qust_pipeline.groovy first"
    return
}

def annots = getAnnotationObjects()
if (annots.isEmpty()) {
    println "  [skip] no tissue annotation -- run_qust_pipeline.groovy must be re-run"
    return
}
println "  selecting ${annots.size()} tissue annotation(s)"
selectAnnotations()

println "  XeniumAnnotation ..."
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

def nAfter = getDetectionObjects().size()
def tally = [:].withDefault { 0 }
def unlabeled = 0
getDetectionObjects().each {
    def pc = it.getPathClass()
    if (pc == null) unlabeled += 1
    else tally[pc.getName()] += 1
}
println "  detections after:  ${nAfter}  (delta: ${nAfter - nBefore})"
println "  unique cluster ids: ${tally.size()}"
println "  cluster_id tally (top 10):"
tally.sort { -it.value }.take(10).each { k, v -> println "    ${k}: ${v}" }
if (unlabeled > 0) println "    (unlabeled): ${unlabeled}"

// Sanity: cluster ids should be small integers, not pantissue label strings
def sampleNames = tally.keySet().take(5) as List
def looksNumeric = sampleNames.every { it ==~ /\d+/ }
println "  PathClass names look numeric (cluster_id form): ${looksNumeric}"
if (!looksNumeric) {
    println "  WARNING: PathClasses are not cluster_ids -- XeniumAnnotation may have failed"
}
println "done."
