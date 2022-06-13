function bbox(coords) {
    const x = coords.map(c => c[0])
    const y = coords.map(c => c[1])
    return [
        [Math.min(...x), Math.min(...y)],
        [Math.max(...x), Math.max(...y)],
    ]
}

export {bbox}
