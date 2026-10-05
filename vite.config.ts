import { readFileSync } from 'node:fs';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [{ name: 'licences', generateBundle() {
    for (const fileName of ['LICENSE', 'THIRD_PARTY_NOTICES.md']) this.emitFile({ type: 'asset', fileName, source: readFileSync(fileName, 'utf8') });
  } }],
  build: { target: 'es2022' },
});
