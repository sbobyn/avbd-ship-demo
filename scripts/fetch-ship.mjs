// Restore the pinned, CC0 Poly Haven glTF and verify every downloaded file.
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = new URL('../public/ship/', import.meta.url);
const manifest = JSON.parse(await readFile(new URL('manifest.json', root), 'utf8'));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
for (const file of manifest.files) {
  const dest = new URL(file.path, root);
  const existing = await readFile(dest).catch(() => null);
  if (existing && hash(existing) === file.sha256) continue;
  const response = await fetch(file.url);
  if (!response.ok) throw new Error(`${file.path}: HTTP ${response.status}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  if (hash(bytes) !== file.sha256) throw new Error(`${file.path}: checksum mismatch; leaving local file untouched`);
  await mkdir(dirname(fileURLToPath(dest)), { recursive: true });
  await writeFile(dest, bytes);
  console.log(`Restored ${file.path}`);
}
console.log(`Verified ${manifest.files.length} ship assets.`);
