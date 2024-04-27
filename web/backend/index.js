const express = require('express')
const {exec} = require('child_process')
const app = express()
const fs = require('fs');
const os = require('os');
const path = require('path');

// serve the client
app.use(express.static('../frontend'))
// allow express to parse application/json
app.use(express.json()); 

app.post('/compile', (request, response) => {
    const body = request.body

    // Generate a temp file path
    const tempFilePath = path.join(os.tmpdir(), 'tempfile.txt');
    // Write "hello" to the temp file
    fs.writeFile(tempFilePath, body.source, (err) => {
        if (err) {
            response.status(500).send('Failed to write to temp file');
            return;
        }
        response.send(`File created and written to successfully: ${tempFilePath}`);
    });
    const command = "../../build/lettucec " + tempFilePath
    exec(command, (error, stdout, stderr) => {
        const compileResult = {
            "stdout": stdout,
            "stderr": stderr,
            "retcode": "todo" 
        }
        // exec(./, (error, stdout, stderr) => {
        //     const compileResult = {
        //         "stdout": stdout,
        //         "stderr": stderr,
        //         "retcode": "todo" 
        //     }
        //     console.log(`stdout: ${stdout}`);
        //     response.json(compileResult)
        // });
        response.json(compileResult)
    });
})

const PORT = 3001
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`)
})