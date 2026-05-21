/**
 * load_mapping.groovy
 * -------------------
 * Replace the PathClass on every detection in the current image, using a
 * 2-column CSV: first col = current PathClass name (typically a Xenium
 * cluster id as written by QuST's XeniumAnnotation), second col = new
 * PathClass name (a pantissue / hne label string).
 *
 * Usage:
 *   CLI batch (headless, every image in the project):
 *     QuPath script -s -p data/qprj/project.qpproj \
 *         -a /path/to/celltype_assignment_pantissue_label.csv \
 *         cellvit-training/qupath/load_mapping.groovy
 *     # -s persists the new PathClasses into each image's .qpdata
 *
 *   GUI: just run the script and pick the CSV when prompted.
 */
import java.io.File
import java.io.BufferedReader
import java.io.FileReader

// CSV path: prefer args[0] (CLI), fall back to file picker (GUI)
def csvFile = null
if (binding.hasVariable('args') && args != null && args.length > 0) {
    csvFile = new File((String) args[0])
    println "Using CSV from args[0]: ${csvFile.getAbsolutePath()}"
} else {
    csvFile = qupath.lib.gui.dialogs.Dialogs.promptForFile("Select CSV file", null, "CSV files", ".csv")
    if (csvFile == null) {
        println "No file selected, exiting..."
        return
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
} catch (Exception e) {
    println "Error processing CSV file: ${e.getMessage()}"
    e.printStackTrace()
}

