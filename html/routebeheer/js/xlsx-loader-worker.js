/**
 * A Worker() to load XLSX data in the background.
 *
 * Communication works like this:
 * 1. front -> worker: {workbook: '<url>', sheets: [sheet names...]}
 * 2. worker -> front: {workbook: '<url>', sheets: [sheet names...], response: [{row data, sheet: sheet name}, ...]}
 */
importScripts('./xlsx.mini-0.18.8.min.js')

onmessage = async function(event) {
    const {workbook, sheets} = event.data

    const res = await fetch(workbook, {cache: 'no-cache'})
    const buffer = await res.arrayBuffer()
    const book = XLSX.read(buffer, {
        raw: true,
        cellFormula: false,
        cellHTML: false,
        cellText: false,
        sheets: sheets,
    })

    // Concatenate all sheets into one data array.
    // Add the sheet name to each record.
    const data = Object.fromEntries(sheets.map(sheet => {
        const rows = XLSX.utils.sheet_to_json(book.Sheets[sheet], {header: 1})
        return [sheet, {header: rows[0], rows: rows.slice(1)}]
    }))

    postMessage({...event.data, response: data})
}
