# Input CSV schema

Provide one or more CSV files per input folder, with columns:

~~~csv
t,x,y,u
0.000000,0.100000,0.200000,0.012000
0.500000,0.100000,0.200000,-0.008000
~~~

This illustrates the schema only; it is not a training dataset.

Coordinates t, x, y must already be normalized to [0,1] using a shared physical time/space convention across healthy, damaged, and optional simulation input folders. Normalize displacement u using one common amplitude scale. Keep physical units and preprocessing metadata with your data.

Configure the corresponding physical spans in src/config.py before training. See the complete [data contract](../../docs/quickstart.md#data-contract).

The original simulation and experimental data are not included. Generate the analytic healthy-field execution fixture with:

~~~bash
python scripts/generate_demo.py --output data/demo
~~~
