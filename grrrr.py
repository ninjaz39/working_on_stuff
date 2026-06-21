import imageio.v2 as imageio

reader = imageio.get_reader("ak.mp4")
fps = reader.get_meta_data()["fps"]

# 2× slower
slowdown_factor = 5

writer = imageio.get_writer(
    "ak_slow.mp4",
    fps=fps / slowdown_factor
)

for frame in reader:
    writer.append_data(frame)

writer.close()
reader.close()