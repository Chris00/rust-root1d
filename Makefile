TIME = time --format "    %Uuser %Ssystem %Eelapsed %PCPU %Mk"
TMP ?= /tmp

build doc:
	cargo $@ --all-features

examples:
	cargo run --features="rug" --example basic

bench:
	@echo "Dune build"
	-@$(TIME) dune build examples/speed.exe
	-@$(TIME) _build/default/examples/speed.exe
	@$(TIME) cargo build --profile release \
	  --example speed --example speed_toms748 --example speed_f64 \
	  --example speed_roots
	@$(TIME) target/release/examples/speed_f64
	@$(TIME) target/release/examples/speed_roots
	@$(TIME) target/release/examples/speed
	@$(TIME) target/release/examples/speed_toms748

bench_mut:
	@$(TIME) cargo build --profile release --features rug \
	  --example speed_rug_bisect --example speed_rug_toms748
	@$(TIME) target/release/examples/speed_rug_bisect
	@$(TIME) target/release/examples/speed_rug_toms748

flamegraph:
	cargo build --profile release --example speed
	flamegraph -o $(TMP)/speed.svg -- target/release/examples/speed


clean:
	cargo clean
	-dune clean

.PHONY: build doc examples bench bench_mut flamegraph clean
