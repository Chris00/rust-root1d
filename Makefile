TIME = time --format "%Uuser %Ssystem %Eelapsed %PCPU %Mk"
TMP ?= /tmp

build doc:
	cargo $@ --features rug

examples:
	cargo run --features="rug" --example basic

bench:
	-@$(TIME) dune build examples/speed.exe
	-@$(TIME) _build/default/examples/speed.exe
	@$(TIME) cargo build --profile release --features rug \
	  --example speed --example speed_f64 --example speed_rug
	@$(TIME) target/release/examples/speed_rug
	@$(TIME) target/release/examples/speed
	@$(TIME) target/release/examples/speed_f64

flamegraph:
	cargo build --profile release --example speed
	flamegraph -o $(TMP)/speed.svg -- target/release/examples/speed


clean:
	cargo clean
	-dune clean

.PHONY: build doc examples bench flamegraph clean
