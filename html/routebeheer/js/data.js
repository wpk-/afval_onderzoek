const bronnen = {
    buurten: '/data/buurten-2022-06-07.json',
    containers: '/data/containers-2022-06-07.json',
    ggwgebieden: '/data/ggwgebieden-2022-06-07.json',
    routes: '/data/routes-2022-05-12.xlsx',
    stadsdelen: '/data/stadsdelen-2022-06-07.json',
    tiles: 'https://t1.data.amsterdam.nl/topo_wm/{z}/{x}/{y}.png',
}
const fracties = ['glas', 'papier', 'rest', 'rest zuidoost', 'rest noord ois', 'textiel']

const xlsxWorker = new Worker('/js/xlsx-loader-worker.js')
xlsxWorker.addEventListener('error', event => console.log(event))


async function load(endpoint, ...args) {
    if (!bronnen.hasOwnProperty(endpoint)) {
        // `endpoint` may be the data itself. Then load(data) -> data.
        return endpoint
    }

    const path = bronnen[endpoint]
    const ext = path.toLowerCase().replace(/.*(\.[^.]+)$/, '$1')

    const loader = {
        '.json': loadJSON,
        '.xlsx': loadXLSX,
    }

    return loader[ext](path, ...args)
}

async function loadJSON(url) {
    const res = await fetch(url)
    const json = await res.json()
    return parseArrayData(json)
}

function loadXLSX(url, sheets) {
    return new Promise((resolve, reject) => {
        const x = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER)
        xlsxWorker.addEventListener('message', handler)
        xlsxWorker.postMessage({workbook: url, sheets, callId: x})

        function handler(event) {
            const {callId, error, response} = event.data
            if (callId === x) {
                xlsxWorker.removeEventListener('message', handler)
                if (error) {
                    reject(error)
                }
                else {
                    resolve(Object.fromEntries(Object.entries(response).map(
                        ([k, v]) => [k, parseArrayData(v)])))
                }
            }
        }
    })
}

function parseArrayData({header, rows}) {
    return rows.map(
        row => Object.fromEntries(header.map((field, i) => [field || '', row[i]])))
}

export {bronnen, fracties, load}
