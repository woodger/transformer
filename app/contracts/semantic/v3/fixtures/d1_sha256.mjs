#!/usr/bin/env node

import { readFile } from "node:fs/promises";

import { canonicalize, sha256 } from "./jcs.mjs";

const [fixturePath] = process.argv.slice(2);
if (fixturePath === undefined) {
  throw new TypeError("usage: d1_sha256.mjs <fixture.json>");
}

const fixture = JSON.parse(await readFile(fixturePath, "utf8"));
const modelContract = fixture.modelContract;
const objectiveLanguageRevision = 3;
const targetPreimage = {
  objectiveLanguageRevision,
  targetContract: modelContract.targetContract,
};
const objectivePreimage = {
  objectiveLanguageRevision,
  objective: modelContract.objective,
};
const targetContractSha256 = sha256(targetPreimage);
const objectiveSha256 = sha256(objectivePreimage);
process.stdout.write(`${JSON.stringify({
  canonicalJson: {
    targetContract: canonicalize(targetPreimage),
    objective: canonicalize(objectivePreimage),
  },
  targetContractSha256,
  objectiveSha256,
})}\n`);
