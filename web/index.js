const express = require('express')
const app = express()
const port = 8888

app.get('/', (req, res) => {
  return res.redirect('/index.html')
})

// get this directory we're in
const path = require('path')
const dir = path.dirname(__filename)

// get ffml.h
const fs = require('fs')
const ffml_h = fs.readFileSync(dir + '/../src/ffml/ffml.h').toString()
const after = ffml_h.split('enum ffml_op_type')[1];
const after_lines = after.split('\n');
// ignore the remainder of the first line
const after_newline = after_lines.slice(1).join('\n');
const before = after_newline.split('}')[0];

// get the ops
let ops = []

// never do this in production, obv. security
for(let line of before.split('\n')) {
  line = line.trim();
  if(line.startsWith('//')) continue;
  
  // try to get the first word
  const first_word = line.split(' ')[0];

  // if starts with FFML_OP_
  if(first_word.startsWith('FFML_OP_')) {
    // get the name
    let name = first_word;

    name = name.replace(',', '');
    name = name.trim();

    // add it to the list
    ops.push(name);
  }
}


// print it
// console.log(ops)

// serve it
app.get('/ops', (req, res) => {
  return res.json(ops)
})

// static
app.use(express.static(__dirname + '/public'))

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`)
});