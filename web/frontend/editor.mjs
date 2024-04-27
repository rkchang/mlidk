import {EditorView, basicSetup} from "codemirror"
import {javascript} from "@codemirror/lang-javascript"

let editor = new EditorView({
  extensions: [basicSetup],
  parent: document.getElementById("editor") 
})

async function postEditorState() {
  let payload = {
    source: editor.state.doc.toString()
  }
  const response = await fetch("http://127.0.0.1:3001/compile", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  })
  console.log(response)
  const json = await response.json();
  const output =`
    Compile retcode: ${json.compile_retcode}\n
    Compile stdout: ${json.compile_stdout}\n
    Compile stderr: ${json.compile_stderr}\n
    Exec retcode: ${json.exec_retcode}\n
    Exec stdout: ${json.exec_stdout}\n
    Exec stderr: ${json.exec_stderr}\n
    `
  document.getElementById("result").innerText = output;
}

editor.dom.addEventListener("keyup", (event) => {
  postEditorState();
});

