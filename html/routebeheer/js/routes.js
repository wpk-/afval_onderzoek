import {BaseControl, SelectionControl, RadioSelect, MultiSelectMenu, cascade
} from './controls.js'


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


function routeData(data) {
    const stringSort = (a, b) => a > b ? 1 : (b > a ? -1 : 0)

    const alleFracties = Object.keys(data)
    const alleDagen = [...new Set([].concat(...
        Object.entries(data).map(([fractie, rows]) =>
            inzameldagen(Object.keys(rows[0])).map(dag => ({fractie, dag}))
        )
    ))]
    const alleAdressen = [].concat(...Object.entries(data)
        .map(([fractie, rows]) => rows.map(r => {r.fractie = fractie; return r})))
        .sort((a, b) => stringSort(a.Chauffeursomschrijving, b.Chauffeursomschrijving))

    const routes = {}

    Object.entries(data).forEach(([fractie, rows]) => {
        const header = Object.keys(rows[0])
        const dagHeaders = inzameldagen(header)
        const posHeaders = dagHeaders.map(d => `RouteNrs.${d.split('.')[1]}`)

        dagHeaders.forEach((dag, i) => {
            const pos = posHeaders[i]

            rows.forEach(row => {
                if (row[dag] && row[dag].trim()) {
                    const routeId = `${fractie}: ${row[dag]}, ${dag}`

                    if (routes.hasOwnProperty(routeId)) {
                        routes[routeId].geo.push([row[pos], [row.Lat, row.Long]])
                    }
                    else {
                        routes[routeId] = {
                            routeId,
                            fractie,
                            dag,
                            route: row[dag],
                            geo: [[row[pos], [row.Lat, row.Long]]]
                        }
                    }
                }
            })
        })
    })

    const alleRoutes = Object.values(routes)
    .map(r => {r.geo = r.geo.sort((a, b) => a[0] - b[0]).map(([_, g]) => g); return r})
    .sort((a, b) => stringSort(a.route, b.route))
    // [
    //    {
    //        routeId: 'rest: OR Zuid 1, Maandag.1',
    //        fractie: 'rest',
    //        dag: 'Maandag.1',
    //        route: 'OR Zuid 1',
    //        geo: [[lat, long], ...],
    //    },
    //    ...
    // ]

    return {alleFracties, alleDagen, alleRoutes, alleAdressen}
}

class RouteSelectie extends BaseControl {
    constructor(targets, props) {
        super(targets, props)
        // {
        //  data: {
        //      'glas': [[...], ...],
        //      'papier': [[...], ...],
        //      'rest': [[...], ...],
        //      ...
        //  },
        //  getLabel: (d) => ...,
        //  onChange: (evt) => ...,
        // }
    }

    get fractie() {
        // Target 0 is the radio selection.
        const menu = this.target[0]
        return menu.selected
    }

    get layer() {

    }

    render() {
        const data = this.props.data

        // Target 0 selects the "fractie" (radio menu).
        this.target[0].setProps({
            data: Object.keys(data),
            getLabel: d => d,
        })

        // Target 1 selects the day (select menu).
        const dagen = []
        Object.values(data).forEach(({header}) => {
            inzameldagen(header).forEach(dag => {
                if (dagen.indexOf(dag) < 0) {
                    dagen.push(dag)
                }
            })
        })
        this.target[1].setProps({
            data: dagen,
            getLabel: d => d,
        })
    }

    setProps({data, ...props}) {
        super.setProps(props)

        const {
            alleFracties,
            alleDagen,
            alleRoutes,
            alleAdressen,
        } = routeData(data)

        this.target[0].setProps({
            data: alleFracties,
            getLabel: d => d,
            getId: d => d,
            onChange: () => {
                cascade(this.target[0], this.target[1])
                cascade(this.target[0], this.target[3])
            },
        })
        this.target[1].setProps({
            data: alleDagen,
            getLabel: d => d.dag,
            getId: d => `${d.fractie}: ${d.dag}`,
            getParent: d => d.fractie,
            onChange: () => cascade(this.target[1], this.target[2]),
        })
        this.target[2].setProps({
            data: alleRoutes,
            getLabel: d => `${d.route} — ${d.dag.toLowerCase().substr(0, 2)}.${d.dag.split('.')[1]}`,
            getId: d => d.routeId,
            getParent: d => `${d.fractie}: ${d.dag}`,
            //onChange: this._onChange,
        })
        this.target[3].setProps({
            data: alleAdressen,
            getLabel: d => d.Chauffeursomschrijving,
            getParent: d => d.fractie,
            onChange: this._onChange,
        })
    }
}

function inzameldagen(header) {
    const week = new Set(['maandag', 'dinsdag', 'woensdag', 'donderdag',
                          'vrijdag', 'zaterdag', 'zondag'])
    return header.filter(val => week.has(val.toLowerCase().split('.')[0]))
}

export {RouteSelectie, routeData}
