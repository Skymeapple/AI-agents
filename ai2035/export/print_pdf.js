// Print an HTML file to A4 PDF with Chromium. Usage: node print_pdf.js in.html out.pdf
const { chromium } = require('playwright');
(async () => {
  const [inp, out] = process.argv.slice(2);
  const b = await chromium.launch();
  const pg = await b.newPage();
  await pg.goto('file://' + require('path').resolve(inp), { waitUntil: 'load' });
  await pg.pdf({
    path: out, format: 'A4', printBackground: true, displayHeaderFooter: true, outline: true, tagged: true,
    headerTemplate: '<span></span>',
    footerTemplate: '<div style="font-size:8pt;width:100%;text-align:center;color:#666;"><span class="pageNumber"></span> / <span class="totalPages"></span></div>',
    margin: { top: '18mm', bottom: '18mm', left: '16mm', right: '16mm' },
  });
  await b.close();
})();
