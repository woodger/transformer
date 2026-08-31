#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

const [fixturePath] = process.argv.slice(2);
if (fixturePath === undefined) {
  throw new TypeError("usage: event_id_sha256.mjs <identity.json>");
}

const identity = JSON.parse(await readFile(fixturePath, "utf8"));
const canonical = JSON.stringify(identity);
const digest = createHash("sha256").update(canonical, "utf8").digest("hex");
process.stdout.write(`${digest}\n`);
