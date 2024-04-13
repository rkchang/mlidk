bundle:
	./node_modules/.bin/rollup editor.mjs -c # Bundle all our js together into editor.bundle.js

watch:
	while inotifywait -e close_write editor.mjs; do make; done
