import {load} from './data.js'
import {bbox} from './geometry.js'
import {BaseControl, MultiSelectMenu, cascade} from './controls.js'


class GebiedenSelectie extends BaseControl {
    static amsterdamBox = [
        [4.29, 52.71],
        [5.6237, 52.71],
        [5.6237, 52.037],
        [4.29, 52.037],
    ]

    constructor(targets, props) {
        super(targets, props)

        targets.forEach((target, i, _targets) => {
            const next = i + 1
            target.setProps({onChange: next >= _targets.length
                ? this._onChange
                : () => cascade(target, _targets[next])})
        }, this)
    }

    get layer() {
        const coords = this.selected.map(d => d.geo)

        return new window.deck.PolygonLayer({
            id: 'gebied',
            data: [[this.constructor.amsterdamBox, ...coords]],
            pickable: false,
            stroked: true,
            filled: true,
            lineWidthMinPixels: 1,
            getPolygon: d => d,
            getFillColor: [0, 0, 0, 50],
        })
    }

    get selected() {
        const menu = this.target
        .findLast(m => m.selectedIndex > -1) || this.target[0]

        return menu.selected
    }

    render() {
        this.targets.forEach(target => target.render())
    }
}

export {GebiedenSelectie}
