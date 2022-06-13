
class BaseControl {
    constructor(target, props) {
        this.target = typeof target === 'string'
            ? document.querySelector(target) : target
        this.props = {}
        // {
        //  name: '...',
        //  data: [...],
        //  getLabel: (d) => ...,
        //  onChange: (evt) => ...,
        // }

        this._onChange = this._onChange.bind(this)
        //this.render()
        this.setProps(props)
    }

    _onChange(event) {
        if (this.props.hasOwnProperty('onChange')) {
            this.props.onChange.call(null, event)
        }
    }

    render() {}

    setProps(props) {
        Object.assign(this.props, props)
    }
}

class SelectionControl extends BaseControl {
    get selected() {
        const data = this.props.data
        return data[this.selectedIndex]
    }

    get selectedIndex() {}
    set selectedIndex(index) {}

    setEnabled(fn) {}

    setProps(props) {
        super.setProps(props)
        if (props.hasOwnProperty('data') || props.hasOwnProperty('getLabel')) {
            this.render()
        }
    }
}

class RadioSelect extends SelectionControl {
    constructor(target, props) {
        super(target, props)

        this.options = []
        this.render()
    }

    get selectedIndex() {
        return this.options.findIndex(input => input.checked)
    }

    set selectedIndex(index) {
        this.options[index].checked = true
    }

    render() {
        const data = this.props.data || []
        const name = this.props.name
        const getLabel = this.props.getLabel
        const html = (d) => `<label><input type="radio" name="${name}">` +
                            ` <span>${getLabel(d)}</span></label>`

        this.target.innerHTML = data.map(html).join('')

        this.options = Array.from(this.target.querySelectorAll('input'))
        this.options[0].checked = true

        this.options.forEach(
            elm => elm.addEventListener('change', this._onChange))
    }

    setEnabled(fn) {
        const data = this.props.data

        this.options.forEach((option, i) => {
            if ((option.disabled = !fn(data[i], i)) && option.checked) {
                option.checked = false
            }
        })
    }
}

class MultiSelectMenu extends SelectionControl {
    constructor(target, props) {
        super(target, props)

        this.render()
        this.target.addEventListener('change', this._onChange)
    }

    get selected() {
        const data = this.props.data

        if (this.target.selectedIndex === 0) {
            const options = this.target.options
            return data.filter((_, i) => !options[i+1].disabled)
        }
        else {
            return Array.from(this.target.selectedOptions)
                .map((option) => data[option.index-1])
        }
    }

    get selectedIndex() {
        return this.target.selectedIndex - 1
    }

    set selectedIndex(value) {
        this.target.selectedIndex = value + 1
        this.behave()
    }

    _onChange(event) {
        this.behave()
        super._onChange(event)
    }

    behave() {
        const menu = this.target
        if (menu.selectedIndex > 0) {
            menu.multiple = true
        } else {
            menu.selectedIndex = 0
            menu.multiple = false
        }
    }

    render() {
        const data = this.props.data || []
        const getLabel = this.props.getLabel
        const html = (d) => `<option>${getLabel(d)}</option>`

        this.target.innerHTML =
            '<option selected>[ alle ]</option>' + data.map(html).join('')
        this.behave()
    }

    setEnabled(fn) {
        const data = this.props.data
        Array.from(this.target.options)
        .slice(1)
        .forEach((option, i) => {
            if ((option.disabled = !fn(data[i], i)) && option.selected) {
                option.selected = false
            }
        })
        this.behave()
    }
}

function cascade(menu_from, menu_to) {
    const getId = menu_from.props.getId
    const selected = menu_from.selected
    const x = new Set(Array.isArray(selected)
                      ? selected.map(getId) : [getId(selected)])

    const getParent = menu_to.props.getParent
    menu_to.setEnabled(d => x.has(getParent(d)))
    menu_to.selectedIndex = -1     // Select all.

    if (menu_to.props.hasOwnProperty('onChange')) {
        menu_to.props.onChange()
    }
}

export {BaseControl, SelectionControl, RadioSelect, MultiSelectMenu, cascade}
