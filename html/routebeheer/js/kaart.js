import {load, bronnen as sources} from './data.js'

const lijnkleur = {
    'Rest': [100, 100, 100],
    'Glas': [230, 230, 80],
    'Papier': [80, 210, 250],
    'GFT': [0, 180, 40],
    'Textiel': [230, 80, 230],
}

const vulkleur = {}
for (const [k, [r, g, b]] of Object.entries(lijnkleur)) {
    vulkleur[k] = [r, g, b, 50]
}


function amsterdamLayer() {
    const deck = window.deck

    return new deck.TileLayer({
        id: 'amsterdam',
        data: sources['tiles'],

        minZoom: 10,
        //maxZoom: 21,
        tileSize: 256,

        renderSubLayers: props => {
            const {
                bbox: {west, south, east, north}
            } = props.tile;

            return new deck.BitmapLayer(props, {
                data: null,
                image: props.data,
                bounds: [west, south, east, north]
            });
        }
    })
}

function containersLayer(data) {
    const deck = window.deck

    return new deck.ScatterplotLayer({
        id: 'containers',
        data: data || load('containers'),
        getPosition: d => d.geo,
        getFillColor: d => vulkleur[d.fractie],
        getLineColor: d => lijnkleur[d.fractie],
        getRadius: 2,
        lineWidthMinPixels: 2,
        filled: true,
        stroked: true,
    })
}

export {amsterdamLayer, containersLayer}
