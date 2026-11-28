// Render paper.html to paper.pdf with the preinstalled Playwright Chromium.
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage();
  await p.goto('file://' + path.resolve(__dirname, 'paper.html'), { waitUntil: 'load' });
  await p.pdf({ path: path.resolve(__dirname, 'paper.pdf'), format: 'A4', printBackground: true,
    margin: { top: '18mm', bottom: '18mm', left: '16mm', right: '16mm' }, displayHeaderFooter: true,
    headerTemplate: '<span></span>',
    footerTemplate: '<div style="font-size:8px;width:100%;text-align:center;color:#777"><span class="pageNumber"></span></div>' });
  await b.close();
})();
