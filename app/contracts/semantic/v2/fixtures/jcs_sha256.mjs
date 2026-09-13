#!/usr/bin/env node

import { readFile } from "node:fs/promises";

import { canonicalize, sha256 } from "./jcs.mjs";

const [documentPath, property] = process.argv.slice(2);
if (documentPath === undefined) {
  throw new TypeError("usage: jcs_sha256.mjs <document.json> [property]");
}

const document = JSON.parse(await readFile(documentPath, "utf8"));
const value = property === undefined ? document : document[property];
if (value === undefined) {
  throw new TypeError(`missing document property: ${property}`);
}

process.stdout.write(`${JSON.stringify({
  canonicalJson: canonicalize(value),
  sha256: sha256(value),
})}\n`);
