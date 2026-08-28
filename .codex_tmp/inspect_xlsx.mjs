import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const paths = process.argv.slice(2);
for (const path of paths) {
  const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(path));
  const out = await wb.inspect({
    kind: "workbook,sheet,table,region",
    maxChars: 20000,
    tableMaxRows: 80,
    tableMaxCols: 30,
    tableMaxCellChars: 200,
  });
  console.log(`\n===== ${path} =====\n${out.ndjson}`);
}
