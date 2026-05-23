/**
 * load_mapping.groovy
 * -------------------
 * Replace the PathClass on every detection in the current image, using a
 * 2-column CSV: first col = current PathClass name (typically a Xenium
 * cluster id as written by QuST's XeniumAnnotation), second col = new
 * PathClass name (a pantissue / hne label string).
 *
 * CSV resolution order (first match wins):
 *   1. args[0]               -- explicit path (single-image use)
 *   2. <image-parent>/outs/celltype_assignment_pantissue_label.csv
 *      (CLI batch mode: one Xenium outs/ per H&E image)
 *   3. GUI file picker       -- fallback when running interactively
 *
 * Usage:
 *   CLI batch (headless, every image in the project, auto-resolves CSV
 *   from each image's outs/ dir):
 *     QuPath script -s -p data/qprj/project.qpproj \
 *         cellvit-training/qupath/load_mapping.groovy
 *
 *   CLI single image with explicit CSV:
 *     QuPath script -s -p data/qprj/project.qpproj \
 *         -i <image-name> \
 *         -a /path/to/celltype_assignment_pantissue_label.csv \
 *         cellvit-training/qupath/load_mapping.groovy
 *
 *   GUI: run the script and pick the CSV when prompted.
 *
 * `-s` persists the new PathClasses into each image's .qpdata.
 */
import java.io.File
import java.io.BufferedReader
import java.io.FileReader
import java.nio.file.Paths

// CSV path: prefer args[0], else derive from image outs/, else GUI prompt
def csvFile = null
if (binding.hasVariable('args') && args != null && args.length > 0) {
    csvFile = new File((String) args[0])
    println "Using CSV from args[0]: ${csvFile.getAbsolutePath()}"
} else {
    // Try to auto-resolve from current image's outs/ dir
    def imageData = getCurrentImageData()
    if (imageData != null) {
        def server = imageData.getServer()
        def imgName = server.getMetadata().getName()
        def uri = server.getURIs().iterator().next()
        def imgPath = Paths.get(uri)
        def candidate = imgPath.getParent()
                              .resolve("outs")
                              .resolve("celltype_assignment_pantissue_label.csv")
                              .toFile()
        if (candidate.exists()) {
            csvFile = candidate
            println "==== ${imgName} ===="
            println "  Using CSV from image outs/: ${csvFile.getAbsolutePath()}"
        } else {
            println "==== ${imgName} ===="
            println "  [skip] no CSV at: ${candidate.getAbsolutePath()}"
            return
        }
    } else {
        csvFile = qupath.lib.gui.dialogs.Dialogs.promptForFile("Select CSV file", null, "CSV files", ".csv")
        if (csvFile == null) {
            println "No file selected, exiting..."
            return
        }
    }
}

// Verify the file exists
if (!csvFile.exists()) {
    println "Error: File does not exist: ${csvFile.getAbsolutePath()}"
    return
}
// Create counters to track progress
def lineCount = 0
def processedCount = 0
// Track CSV structure
def headers = []
def columnCount = 0
try {
    // Open the file for reading
    BufferedReader reader = new BufferedReader(new FileReader(csvFile))
    String line
    // Read the file line by line
    while ((line = reader.readLine()) != null) {
        lineCount++
        // Handle the header row (first line)
        if (lineCount == 1) {
            // Parse headers
            headers = line.split(",")
            columnCount = headers.length
            println "CSV Headers: ${headers.join(', ')}"
            println "Found ${columnCount} columns"
            continue  
            // Skip to next line
        }
        // Process data rows
        def values = line.split(",")
        // Check if the row has the expected number of columns
        if (values.length != columnCount) {
            println "Warning: Line ${lineCount} has ${values.length} columns (expected ${columnCount})"
        }
        // Process each value in the row
        def rowData = [:]
        for (int i = 0; i < Math.min(values.length, headers.length); i++) {
            // Trim whitespace and quotes from values
            def value = values[i].trim()
            if (value.startsWith('"') && value.endsWith('"')) {
                value = value.substring(1, value.length() - 1)
            }
            rowData[headers[i]] = value
        }
         
        def oldCls = rowData[headers[0]]
        def newCls = getPathClass(rowData[headers[1]])
        
        def objList = getDetectionObjects().findAll { obj ->
            obj.getPathClass() != null && obj.getPathClass().getName() == oldCls
        }
        
        objList.each { obj ->
            obj.setPathClass(newCls)
        }
            processedCount++
    }
    // Close the reader
    reader.close()
    println "CSV processing completed:"
    println "  Total lines: ${lineCount}"
    println "  Processed rows: ${processedCount}"

    // Per-slide tally so batch logs are greppable
    def tally = [:].withDefault { 0 }
    def unlabeled = 0
    getDetectionObjects().each {
        def pc = it.getPathClass()
        if (pc == null) {
            unlabeled += 1
        } else {
            tally[pc.getName()] += 1
        }
    }
    println "  detections labeled: ${getDetectionObjects().size() - unlabeled} / ${getDetectionObjects().size()}"
    println "  class tally:"
    tally.sort { -it.value }.each { k, v -> println "    ${k}: ${v}" }
    if (unlabeled > 0) println "    (unlabeled): ${unlabeled}"
    println "done."
} catch (Exception e) {
    println "Error processing CSV file: ${e.getMessage()}"
    e.printStackTrace()
}

