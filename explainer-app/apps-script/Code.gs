/**
 * Google Apps Script web app for collecting benchmark results.
 *
 * Setup:
 * 1. Open your Google Sheet
 * 2. Extensions → Apps Script
 * 3. Paste this code into Code.gs
 * 4. Update SHEET_NAME below if needed
 * 5. Deploy → New deployment → Web app
 *    - Execute as: Me
 *    - Who has access: Anyone
 * 6. Copy the deployment URL into benchmark.js APPS_SCRIPT_URL
 */

var SHEET_NAME = 'Benchmark v1';

var COLUMNS = [
  'timestamp',
  'userName',
  'machineLabel',
  'os',
  'backend',
  'model',
  'tfjsVersion',
  'gpuRenderer',
  'gpuVendor',
  'cpuCores',
  'deviceMemoryGB',
  'userAgent',
  'platform',
  'screenWidth',
  'screenHeight',
  'devicePixelRatio',
  'windowSeconds',
  'streamDurationSec',
  'numWindows',
  'modelLoadMs',
  'preprocessingMs',
  'inferenceMs',
  'totalMs',
  'realtimeRatio',
  'dataGenerationMs',
];

function getOrCreateSheet() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = ss.getSheetByName(SHEET_NAME);
  if (!sheet) {
    sheet = ss.insertSheet(SHEET_NAME);
  }
  // Add headers if row 1 is empty
  if (sheet.getLastRow() === 0) {
    sheet.getRange(1, 1, 1, COLUMNS.length).setValues([COLUMNS]);
    sheet.getRange(1, 1, 1, COLUMNS.length).setFontWeight('bold');
    sheet.setFrozenRows(1);
  }
  return sheet;
}

function doPost(e) {
  try {
    var data = JSON.parse(e.postData.contents);
    var sheet = getOrCreateSheet();

    // Build row in column order
    var row = COLUMNS.map(function(col) {
      if (col === 'timestamp') {
        return new Date().toISOString();
      }
      return data[col] !== undefined ? data[col] : '';
    });

    sheet.appendRow(row);

    return ContentService
      .createTextOutput(JSON.stringify({ status: 'ok' }))
      .setMimeType(ContentService.MimeType.JSON);
  } catch (err) {
    return ContentService
      .createTextOutput(JSON.stringify({ status: 'error', message: err.message }))
      .setMimeType(ContentService.MimeType.JSON);
  }
}

function doGet() {
  return ContentService
    .createTextOutput(JSON.stringify({ status: 'ok', message: 'Benchmark collector is running.' }))
    .setMimeType(ContentService.MimeType.JSON);
}
