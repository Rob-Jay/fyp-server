const express = require('express')
const exec = require('child_process').exec;
const app = express();
const fs = require('fs')



app.use(express.json());

app.post('/test', (req, res) => {
    res.json({ message: 'great success' })
        //res.send(JSON.stringify("Great Success"))
})


app.post('/action', (req, res) => {
    console.log('Launching python');
    fs.writeFileSync('./Summarizer/Scan.txt', req.body.summarised_text, function(err) {
        if (err) throw err;
        console.log('Saved!');
    });


    const command = `python ./Summarizer/main.py ${req.body.num_sentences}`;
    exec(command, (err, stdout, stderr) => {
        if (err) {
            console.error(`exec error: ${err}`);
            return;
        }

        console.log(JSON.parse(stdout))

        // console.log(`Number of files ${stdout}`);

        res.json({ summarised_text: JSON.parse(stdout).summary, num_sentences: 0 });
    });
})


const port = 5000;
app.listen(port, () => {
    console.log(`listening to port ${port}`);
})