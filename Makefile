TIME = time --format "%Uuser %Ssystem %Eelapsed %PCPU %Mk"

build doc:
	cargo $@ --features rug

examples:
	cargo run --features="rug" --example basic

bench:
	-@$(TIME) dune build examples/speed.exe
	-@$(TIME) _build/default/examples/speed.exe
	@$(TIME) cargo build --profile release \
	  --example speed --example speed_f64
	@$(TIME) target/release/examples/speed
	@$(TIME) target/release/examples/speed_f64

clean:
	cargo clean
	-dune clean

.PHONY: build doc examples bench clean
