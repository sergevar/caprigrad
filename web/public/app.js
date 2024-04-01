const init = async() => {
    const response = await fetch('/ops');
    const ops = await response.json();
    console.log(ops);

    window.ops = ops;
}

let myHistoChart;

window.latest_info = {};

$(() => {

init();

var ctx = document.getElementById('trainingLossChart').getContext('2d');
var chart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [], // This will be filled with the iteration number or absolute time
        datasets: [{
            label: 'Training Loss Progress',
            data: [], // This will be filled with the training loss from server
            borderColor: 'rgb(75, 192, 192)',
            tension: 0.1
        }]
    },
    options: {
        scales: {
            y: {
                beginAtZero: true
            }
        },
        animations: {
            y: {
                duration: 0
            }
        },
    }
});

function addData(chart, label, data) {
    chart.data.labels.push(label);
    chart.data.datasets.forEach((dataset) => {
        dataset.data.push(data);
    });
    chart.update({
        duration: 0,
        lazy: false,
    });
}

////////////////////////

console.log('connecting');
var websocket = new WebSocket('ws://localhost:8889');

let callbacks = {};
let eventCallbacks = {};

const api = async(cmd, data = {}) => {
    const request_id = Math.random().toString(36).substring(7);

    var message = {
        request_id,
        command: cmd,
        data
    }

    console.log('Sending message to server: ', message);

    websocket.send(JSON.stringify(message));

    return new Promise((resolve, reject) => {
        callbacks[request_id] = (data) => {
            resolve(data);
        }
    });
};

window.dataChunks = {};

websocket.addEventListener('message', function (event) {
    const eventData = event.data;

    const req_id = eventData.split('|')[0];
    const mode = eventData.substr(req_id.length + 1, 1);
    const theRest = eventData.substr(req_id.length + 2);

    console.log({req_id, mode, theRest})

    if (mode === 'S' || mode === 'M') {
        if (window.dataChunks[req_id] === undefined) {
            if (mode === 'S') {
                window.dataChunks[req_id] = '';                
            } else {
                return;
            }
        }
        window.dataChunks[req_id] += theRest;
    } else if (mode === 'E') {
        if (window.dataChunks[req_id] === undefined) {
            return;
        }

        console.log(window.dataChunks[req_id]);

        const data = JSON.parse(window.dataChunks[req_id]);

        window.dataChunks[req_id] = undefined;

        console.log('Message from server: ', data);
    
        if (callbacks[req_id]) {
            callbacks[req_id](data);
            delete callbacks[req_id];
        } else {
            // is an event
            let eventType = data.event;
            let eventData = data.data;
    
            console.log('Event: ', eventType, eventData);
    
            if (eventCallbacks[eventType]) {
                eventCallbacks[eventType](eventData);
            }
        }
    }

});

let cgraph_nodes = {};

const visData = (data, key, grad) => {
    // console.log({data}, 'here')
    const cgraph_tensor_data = data.cgraph_tensor_data;
    // console.log({cgraph_tensor_data})
    const tdata = JSON.parse((grad) ? cgraph_tensor_data.grad : cgraph_tensor_data.data);
    const tensor = cgraph_nodes[key];

    let visualizer = new Visualizer();
    
    if (tensor.n_dims == 2) {
        let dim1 = tensor.ne[0];
        let dim2 = tensor.ne[1];
        
        let table = [];
        for (let i = 0; i < dim1; i++) {
            let row = [];
            for (let j = 0; j < dim2; j++) {
                row.push(tdata[i * dim2 + j]);
            }
            table.push(row);
        }

        visualizer.tables.push(table);
    } else if (tensor.n_dims == 1) {
        let table = [];
        let row = [];
        for (let i = 0; i < tensor.ne[0]; i++) {
            row.push(tdata[i]);
        }
        table.push(row);

        visualizer.tables.push(table);
    } else if (tensor.n_dims == 3) {
        let dim1 = tensor.ne[0];
        let dim2 = tensor.ne[1];
        let dim3 = tensor.ne[2];
        
        let tables = [];
        for (let i = 0; i < dim1; i++) {
            let table = [];
            for (let j = 0; j < dim2; j++) {
                let row = [];
                for (let k = 0; k < dim3; k++) {
                    row.push(tdata[i * dim2 * dim3 + j * dim3 + k]);
                }
                table.push(row);
            }
            tables.push(table);
        }

        for (let table of tables) {
            visualizer.tables.push(table);
        }
    } else {
        alert('Unsupported tensor dimensions: ' + tensor.n_dims);
        return;
    }

    visualizer.visualize(grad);
}

const clickedTensorKey = (key) => {
    api('CGRAPH_TENSOR_DATA', {key}).then((data) => {
        visData(data, key, false);
        visData(data, key, true);
    });
    // api('CGRAPH_TENSOR_GRAD', {key}).then((data) => {
    //     visData(data, key, true);
    // });
}

let render_cgraph = () => {
    // let html = '<table>';

    // for(let node_key in cgraph_nodes) {
    //     let node = cgraph_nodes[node_key];

    //     let shape = JSON.stringify(node.ne);

    //     let node_html = `
    //         <tr class="cgraph-tensor-row" data-key="${node.key}">
    //             <td>${node.key}</td>
    //             <td>${node.name}</td>
    //             <td>${shape}</td>
    //         </tr>
    //     `;

    //     html += node_html;
    // }

    // html += '</table>';

    // document.getElementById('cgraph_info').innerHTML = html;

    // create an array with nodes
    var nodes_data = [];
    for (let node_key in cgraph_nodes) {
        let node = cgraph_nodes[node_key];

        let shape = [];
        for (let i = 0; i < node.n_dims; i++) {
            shape.push(node.ne[i]);
        }
        shape = "[" + shape.join(', ') + "]";

        nodes_data.push({
            font: { multi: true },
            id: node.key,
            label: "<b>"+node.name + "</b>\n" + shape + "\n" + ops[node.op].replace("FFML_OP_", "").toLowerCase(),
            shape:'box',

            // tooltip
            title: ops[node.op].replace("FFML_OP_", "").toLowerCase(),
        });
    }

    var nodes = new vis.DataSet(nodes_data);

    var edges_data = [];
    for (let node_key in cgraph_nodes) {
        let node = cgraph_nodes[node_key];

        // src0
        if (node.src0 != -1) {
            edges_data.push({
                from: node.src0,
                to: node.key,
            });
        }

        // src1
        if (node.src1 != -1) {
            edges_data.push({
                from: node.src1,
                to: node.key,
            });
        }
    }

    // create an array with edges
    var edges = new vis.DataSet(edges_data);

    // create a network
    var container = document.getElementById('diagram');
    var data = {
        nodes: nodes,
        edges: edges
    };
    var options = {
        layout: {
            hierarchical: {
                // left to right, nicely
                direction: "LR",
                sortMethod: "directed",
                nodeSpacing: 100,
            }
        }
    };
    
    var network = new vis.Network(container, data, options);

    // add event listeners
    network.on("click", function(params) {
            params.event = "[original event]";
        // alert('Clicked node '+params.nodes);
        if (params.nodes.length > 0) {
            let node_key = params.nodes[0];
            clickedTensorKey(node_key);
        }
    });


}

// Connection opened
websocket.addEventListener('open', function (event) {
    console.log('Connected to server');

    // setInterval(() => {
    //     api('ECHO', 'Hello Server!').then((data) => {
    //         console.log('ECHO', data);
    //     });
    // }, 2000);

    // api('REVERSE', 'Hello Server!').then((data) => {
    //     console.log('REVERSE', data);
    // });

    api('CGRAPH_INFO').then((data) => {

        console.log('CGRAPH_INFO', data);

        let cgraph_info = data.cgraph_info;

        cgraph_nodes = cgraph_info.nodes;

        render_cgraph();
    });
});

class Visualizer {
    tables = [];

    visualize(grad) {
        let html = "";

        for(let table of this.tables) {
            html += '<table class="viz-table">';

            for(let i = 0; i < table.length; i++) {
                let row = table[i];

                html += '<tr>';
    
                for(let j = 0; j < row.length; j++) {
                    let value = table[i][j];
    
                    let value_fixed = (grad) ? value.toFixed(6) : value.toFixed(2);
    
                    let r = 255;
                    let g = 255;
                    let b = 255;
    
                    if (value <= 1 && value >= -1) {
                        // gray. intensity based on value
                        let v = value / 2 + 0.5;

                        r = 255 * v;
                        g = 255 * v;
                        b = 255 * v;
                    
                    //     r = 255;
                    //     g = 255;
                    //     b = 255 * (1 + value);
                    // }

                    // if (value > 0 && value <= 1) {
                    //     r = 255;
                    //     g = 255 * (1 - value);
                    //     b = 255;
                    }
    
                    if (value < -1) {
                        // gray. intensity based on log scale
                        let intensity = Math.log10(-value);
                        intensity = Math.min(1, intensity);
                        intensity = Math.max(0, intensity);

                        intensity = intensity / 2 + 0.5;

                        r = 255 * (intensity);
                        g = 0;
                        b = 255 * (intensity);
                    }
    
                    if (value > 1) {
                        // red
                        let intensity = Math.log10(value);
                        intensity = Math.min(1, intensity);
                        intensity = Math.max(0, intensity);

                        intensity = intensity / 2 + 0.5;

                        r = 255 * intensity;
                        g = 255 * intensity;
                        b = 0;
                    }

                    // text color should be the opposite, for contrast
                    let t_r = 255 - Math.max(Math.min(r, 255), 0);
                    let t_g = 255 - Math.max(Math.min(g, 255), 0);
                    let t_b = 255 - Math.max(Math.min(b, 255), 0);

                    // if all of the colors are close (near grayish), make the text black
                    if (Math.abs(r - t_r) < 30 && Math.abs(g - t_g) < 30 && Math.abs(b - t_b) < 30) {
                        t_r = 0;
                        t_g = 0;
                        t_b = 0;
                    }
    
                    let style = `background-color: rgb(${r}, ${g}, ${b}); color: rgb(${t_r}, ${t_g}, ${t_b});`;
    
                    html += `<td style="${style}" title="${value}">${value_fixed}</td>`;
                }
    
                html += '</tr>';
            }
    
            html += '</table>';
        }

        document.getElementById((grad) ? 'tensor_viz_grad' : 'tensor_viz').innerHTML = html;

        if (!grad) this.renderHistogram();
    }

    renderHistogram() {

        var data = [];
        for(let table of this.tables) {
            for(let row of table) {
                for(let value of row) {
                    data.push(value);
                }
            }
        }

      // Step 2 - Pre-process data

      var minData = Math.min(...data);
      var maxData = Math.max(...data);

      console.log({minData, maxData});

      var range = maxData - minData;
      var numBins = 30; // Change this to change the number of bins
      var binSize = range / numBins;

      var histogramData = Array.from({length: numBins}, () => 0);

      
      // Loop over data
      for (var i = 0; i < data.length; i++) {
        var bin = Math.floor((data[i]-minData) / binSize);
        histogramData[bin] += 1;
      }

      var labels = histogramData.map((_, i) => `${(minData + i * binSize).toFixed(4)} to ${(minData + (i+1) * binSize).toFixed(4)}`);

      console.log({labels, histogramData});

      if (typeof myHistoChart !== "undefined") {
        myHistoChart.destroy();
      }

      var ctx = document.getElementById('histogramChart').getContext('2d');

      // Step 3 - Create Chart
      myHistoChart = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{
            label: 'Histogram of values for tensor',
            data: histogramData,
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1
          }]
        },
        options: {
          scales: {
            y: {
              beginAtZero: true
            }
          }
        }
      });


    }
}

$(document).on('click', '.cgraph-tensor-row', function() {
    let key = $(this).data('key');
    clickedTensorKey(key);
});

$('#btn_pause').click(() => {
    api('PAUSE').then((data) => {
        console.log('PAUSE', data);
    });
});

$('#btn_resume').click(() => {
    api('RESUME').then((data) => {
        console.log('RESUME', data);
    });
});

// Listen for close event
websocket.addEventListener('close', function(event) {
    console.log('Server closed connection: ', event.data);
});

// Listen for connection errors
websocket.addEventListener('error', function(error) {
    console.log('Error happened: ', error.data);
});

const renderLatestInfo = () => {
    const json = JSON.stringify(window.latest_info, null, 2);
    document.getElementById('latest-info').innerHTML = json;
}

// Events
eventCallbacks['training_step'] = (data) => {
    console.log('E training_step', data);

    window.latest_info.training_step = data;
    renderLatestInfo();

    addData(chart, data.step, data.loss);
};





})

