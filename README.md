# windrose
Simple windrose and wind distribution chart generator from raw data and frequency tables

The code revolves around the *WindRose* class although useful generic util functions can be found in the util module and the *Weibull* class is under development.

There *WindRose* can be instanciated by passing the flatfile path, the header names for wind speed and wind direction and optional pandas.read_csv loading parameters.

The class has also 3 methods:
- calc_freq_table
- plot_windrose
- export_frequency_table

### Example code
```sh
from windrose.windrose import WindRose

wr = WindRose(fpath='my_wind_data.csv', 
              ws_field='WS01', wd_field='Dir02', 
              na_values=[999, 999.9])
    print(wr)  # -->  WindRose ["WS01" vs "Dir02"]

    ft = wr.calc_freq_table(ws_bin_step=6, ws_bin_centre=3,
                            wd_bin_step=30, wd_bin_centre=0)  # returns a pandas DataFrame
    print(ft.loc['[0, 6)', '[-15, 15)'])  # -->  0.106143315137

    wr.export_frequency_table('my_frequency_table.csv')     # -->  saves the data as csv

    wr.plot_windrose(overlay_sector=[350, 10, 120, 230],
                     save_fig='my_windrose_plot.png')       # -->  saves the plot as image
```
[Here](https://github.com/gabrielecalvo/windrose/raw/master/docs/my_wind_data.csv "sample source file") you can find the example source data file *my_wind_data.csv*.

The resulting frequency distribution file can be found [here](https://github.com/gabrielecalvo/windrose/raw/master/docs/my_frequency_table.csv "sample frequency distribution output"). 

And the resulting wind rose image looks like this:
![Sample Windrose Plot](https://github.com/gabrielecalvo/windrose/raw/master/docs/my_windrose_plot.png?raw=true)

### Todos
 - Implement the Weibull class
 - Further develop the Song class

License
----
MIT