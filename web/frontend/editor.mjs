import {EditorView, basicSetup} from "codemirror"
import {javascript} from "@codemirror/lang-javascript"

let editor = new EditorView({
  extensions: [basicSetup, javascript()],
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
  document.getElementById("result").innerText = await response.text()
}

editor.dom.addEventListener("keyup", (event) => {
  postEditorState();
});

