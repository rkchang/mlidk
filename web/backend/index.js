const express = require('express')
const app = express()
const util = require('util');
const exec = util.promisify(require('child_process').exec);
const fs = require('fs');
const os = require('os');
const path = require('path');

// serve the client
app.use(express.static('../frontend'))
// allow express to parse application/json
app.use(express.json());


app.post('/compile', async(request, response) => {
    const body = request.body

    console.log(body.source)
    // Generate a temp file path
    const tempFilePath = 'tempfile.txt';
    // TODO: make async
    // Write "hello" to the temp file
    fs.writeFileSync(tempFilePath, body.source, (err) => {
        if (err) {
            response.status(500).send('Failed to write to temp file');
            return;
        }
    });
    const compileResult = {
        "compile_stdout": "",
        "compile_stderr": "",
        "compile_retcode": "empty",
        "exec_stdout": "",
        "exec_stderr": "",
        "exec_retcode": "empty"
    }
    console.log(`Current working directory: ${process.cwd()}`);
    try {
        let { stdout, stderr, error } = await exec(`../../build/lettucec/lettucec ${tempFilePath}`)
        compileResult.compile_stdout = stdout;
        compileResult.compile_stderr = stderr;
        compileResult.compile_retcode = error ? error.code : 0;
        await exec(`clang++ output.o -o output`)
        await exec(`chmod +x output`)
        try {
            let { exec_stdout, exec_stderr, exec_error } = await exec(`./output`)
            compileResult.exec_stdout = exec_stdout;
            compileResult.exec_stderr = exec_stderr;
            compileResult.exec_retcode = exec_error ? exec_error.code : 0;
        } catch(e) {
            // exec returns a rejected promise if exit code is non-zero 
            // program may have still succeeded it's just that the return code is non-zero
            compileResult.exec_stdout = e.stdout;
            compileResult.exec_stderr = e.stderr;
            compileResult.exec_retcode = e ? e.code : 0;
        }
    } catch(e) {
        // exec returns a rejected promise if exit code is non-zero 
        compileResult.compile_stdout = e.stdout;
        compileResult.compile_stderr = e.stderr.replace("libc++abi: terminating due to uncaught exception of type Parser::Error: tempfile.txt:", "");
    }
    response.json(compileResult)
})

const PORT = 3001
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`)
})