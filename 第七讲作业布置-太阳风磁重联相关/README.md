# 作业说明

# 下载PSP数据

数据位于：[spdf.gsfc.nasa.gov/pub/data/psp/](http://spdf.gsfc.nasa.gov/pub/data/psp/)

需要用到的数据包括：

- 磁场数据：[https://spdf.gsfc.nasa.gov/pub/data/psp/fields/l2/](https://spdf.gsfc.nasa.gov/pub/data/psp/fields/l2/)
    - 你可以找到三种文件夹（mag_rtn，mag_rtn_4_per_cycle，mag_rtn_1min）其中数据分别对应不同的时间分辨率，从高到低为mag_rtn > mag_rtn_4_per_cycle > mag_rtn_1min，在代码中对应的标记(mag_type)分别为rtn, 4sa, rtn_1min.

【注意】时间分辨率过高可能会导致程序读取缓慢，在绘制Overview的过程中，推荐使用4sa数据；在仔细观察重联事件的时候，可以使用rtn数据；想要快速画出很多天的时候，可以使用rtn_1min数据。

- 质子数据（密度、温度、速度等）：[https://spdf.gsfc.nasa.gov/pub/data/psp/sweap/spi/l3/spi_sf00_l3_mom/](https://spdf.gsfc.nasa.gov/pub/data/psp/sweap/spi/l3/spi_sf00_l3_mom/)
- 电子投掷角分布数据：[https://spdf.gsfc.nasa.gov/pub/data/psp/sweap/spe/l3/spe_sf0_pad/](https://spdf.gsfc.nasa.gov/pub/data/psp/sweap/spe/l3/spe_sf0_pad/)
- 质子VDF数据：[https://spdf.gsfc.nasa.gov/pub/data/psp/sweap/spi/l2/spi_sf00_8dx32ex8a/](https://spdf.gsfc.nasa.gov/pub/data/psp/sweap/spi/l2/spi_sf00_8dx32ex8a/)
    - 质子VDF数据在程序中是单独读取的。通常你只需要在选择好重联时间段后，单独下载该时间段所在的当天的数据。

【注意】下载后的数据可以全部存放在同一文件夹下，该文件夹路径需要被设置为load_read_psp_data.py中的psp_data_path变量

```jsx
# !!!! CHANGE TO YOUR DATA PATH
psp_data_path = '/Users/ephe/PSP_Data_Analysis/Encounter08/'
```

# 绘制PSP Overview

- 设置load_read_psp_data.py中的psp_data_path变量
- 修改plot_psp_overview.py中USER INPUT部分（主要需要选择时间范围）
- 运行plot_psp_overview.py。在理想情况下，可以得到该时间段内的OVERVIEW图和MVA测试图。

【建议】首先选择PSP近日点前后三天左右的数据画图（使用4sa数据），然后选择Br反转附近的几个小时画图，观察是否存在重联特征。

# 判断是否发生磁重联事件

- 搜索并阅读任意一篇有关psp观测太阳风磁重联事件的文献（如Phan 2020）。在理想情况下，你就学会了如何粗略诊断磁重联事件。
- 搜索并阅读任意一篇有关磁重联Minimum Variance Analysis (MVA)分析的文献或课件。在理想情况下，你就理解了什么是磁重联MVA分析。

# 绘制重联事件内外的二维质子VDF时间序列

- 对于已找到的重联事件，下载该事件当天对应的VDF数据文件。

【建议】选择PSP近日点附近的事件，以免数据缺失或质量不好

- 运行plot_2d_psp_vdf.py。需要修改如下段落中的EXPORT_PATH变量，并且手动绘制图像的时间段和分辨率（即vdftime_full_list）。这是一段比较原始的代码，但是应该也比较易懂。

```jsx
vdftime_full_list = [np.datetime64('2022-12-12T03:00:00') + np.timedelta64(180, 's') * n for n in range(180)]
EXPORT_PATH = 'export/work/tres_hcs_crossings/plot_vdf_movie/E14/'
os.makedirs(EXPORT_PATH, exist_ok=True)
for i in range(180):
    plot_vdf_frame(vdftime_full_list[i], 5, frame='rtn')
    time_str = vdftime_full_list[i].tolist().strftime('%H:%M:%S')
    plt.savefig(EXPORT_PATH + 'HCS1_' + time_str + '_vdf_rtn.png')
    plt.close()
```

【注意】你会发现，绘制质子VDF相关图像的代码和绘制PSP Overview的代码中读取数据、处理数据的方式是不一样的（前者使用pyspedas，后者手动写了各种函数）。这是因为这是两个人写出来的。如果你感兴趣并且充满多余的精力，可以把他们统一成同一套方式。

# 绘制重联时间内外的三维质子VDF

- 从已经画出的二维质子VDF切片时间序列中，挑选出你最感兴趣的时刻，绘制三维质子VDF
- 运行pyvista_gif_make_InSB.py，改变变量vdftime为你挑选的时刻。理想情况下，你就能得到三维VDF的图像。

# 相关程序提供者
吴子祺 (Ziqi WU)\\
段叠 (Die DUAN)\\
何建森 (Jiansen HE)

# 相关参考文献
- (1) Duan, D., He, J., Zhu, X., Zhuo, R., Wu, Z., Nicolaou, G., ... & Horbury, T. S. (2023). Kinetic Features of Alpha Particles in a Pestchek-like Magnetic Reconnection Event in the Solar Wind Observed by Solar Orbiter. The Astrophysical Journal Letters, 952(1), L11.
- (2) Wu, Z., He, J., Duan, D., Zhu, X., Hou, C., Verscharen, D., ... & Louarn, P. (2023). Ion Energization and Thermalization in Magnetic Reconnection Exhaust Region in the Solar Wind. The Astrophysical Journal, 951(2), 98.
- (3) Luo, Q., Duan, D., He, J., Zhu, X., Verscharen, D., Cui, J., & Lai, H. (2023). Statistical Study of Anisotropic Proton Heating in Interplanetary Magnetic Switchbacks Measured by Parker Solar Probe. The Astrophysical Journal Letters, 952(2), L40.
- (4) Jiansen, H., Xingyu, Z., Yajie, C., Chadi, S., Michael, S., Hui, L., ... & Chuanyi, T. (2018). Plasma heating and Alfvénic turbulence enhancement during two steps of energy conversion in magnetic reconnection exhaust region of solar wind. The Astrophysical Journal, 856(2), 148.
