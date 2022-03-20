
build doc:
	cargo $@ --features rug

examples:
	cargo run --features="rug" --example basic


.PHONY: build doc examples
