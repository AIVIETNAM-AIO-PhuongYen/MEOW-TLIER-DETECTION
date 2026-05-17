import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
from pathlib import Path

INLIER_COLOR = '#F5F5DC'
OUTLIER_COLOR = '#A0522D'
ELIP_RING_COLOR = '#FFB6C1'

# INLIER_COLOR = 'red'
# OUTLIER_COLOR = 'blue'
# ELIP_RING_COLOR = 'green'

def draw_outlier_ellipse(ax, x_values, y_values, n_std=3.0, **kwargs):
	"""
	Vẽ một elip theo ma trận hiệp phương sai của các điểm (x_values, y_values).
	1. Tính ma trận hiệp phương sai của các điểm (2x2).
	2. Lấy trị số riêng và vectơ riêng (eigen decomposition) để tìm hướng và độ dài
	  của 2 trục chính của elip.
	3. Kích thước elip theo số độ lệch chuẩn `n_std`

	Tham số:
	- ax: trục matplotlib để vẽ lên.
	- x_values, y_values: mảng số liệu (1D) cho trục x và y.
	- n_std: số lượng độ lệch chuẩn để mở rộng elip.
	- **kwargs: truyền cho Ellipse (edgecolor, linewidth, linestyle ...).
	"""
	# Gộp các điểm thành ma trận N x 2
	points = np.column_stack([x_values, y_values])
	if len(points) < 2:
		# Không đủ điểm để tính hiệp phương sai
		return

	# Tính ma trận hiệp phương sai (2x2)
	cov = np.cov(points, rowvar=False)
	if np.any(np.isnan(cov)):
		return

	# Phân tích trị số/vectơ riêng để tìm trục chính
	eigenvalues, eigenvectors = np.linalg.eigh(cov)
	
	order = eigenvalues.argsort()[::-1]
	eigenvalues = eigenvalues[order]
	eigenvectors = eigenvectors[:, order]

	# Góc của elip (theo độ) từ vectơ riêng lớn nhất
	angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
	# Chiều rộng và chiều cao = 2 * n_std * sqrt(eigenvalues)
	width, height = 2 * n_std * np.sqrt(eigenvalues)
	center = (np.mean(x_values), np.mean(y_values))

	# Tạo elip (không tô màu bên trong, chỉ viền)
	ellipse = Ellipse(center, width=width, height=height, angle=angle, fill=False, **kwargs)
	ax.add_patch(ellipse)


def plot_scatter_with_ellipse(df, x_col='food_weight_g', y_col='sleep_hours', hue_column='is_outlier', elip_std=3.0, inlier_color=INLIER_COLOR, outlier_color=OUTLIER_COLOR, elip_ring_color=ELIP_RING_COLOR, **kwargs):
	"""
	Vẽ scatter 2 chiều cho hai cột `x_col` và `y_col`, phân màu theo `hue_column` (nếu có),
	và thêm elip bao quanh các điểm không phải outlier (is_outlier == 0).

	1. Tách các điểm inlier và outlier (nếu cột `hue_column` có mặt).
	2. Vẽ scatter cho inlier (màu xanh) và outlier (màu đỏ) nếu có.
	3. Tính ma trận hiệp phương sai của các điểm inlier và vẽ elip (theo số std `elip_std`).
    """

	# Lấy dataframe với các cột cần dùng
	plot_df = df[[x_col, y_col] + ([hue_column] if hue_column in df.columns else [])].copy()

	# Mặt nạ xác định inlier (is_outlier == 0). Nếu không có cột hue, tất cả là inlier.
	inlier_mask = plot_df[hue_column] == 0 if hue_column in plot_df.columns else pd.Series(True, index=plot_df.index)
	inlier_points = plot_df.loc[inlier_mask]
	outlier_points = plot_df.loc[~inlier_mask]

	sns.set_theme(style='whitegrid')
	fig, ax = plt.subplots(figsize=(9, 7))

	# Vẽ điểm inlier
	sns.scatterplot(
		data=inlier_points,
		x=x_col,
		y=y_col,
		color=inlier_color,
		s=45,
		alpha=0.75,
		label='Non-outlier',
		ax=ax,
	)

	# Nếu có outlier thì vẽ thêm
	if not outlier_points.empty:
		sns.scatterplot(
			data=outlier_points,
			x=x_col,
			y=y_col,
			color=outlier_color,
			s=45,
			alpha=0.85,
			label='Outlier',
			ax=ax,
		)

	# Thêm elip theo hiệp phương sai của inlier
	draw_outlier_ellipse(
		ax,
		inlier_points[x_col].to_numpy(),
		inlier_points[y_col].to_numpy(),
		n_std=elip_std,
		edgecolor=elip_ring_color,
		linewidth=2,
		linestyle='--',
		label='Elip boundary',
	)

	ax.set_title('Scatter plot với elip inlier')
	ax.set_xlabel(x_col)
	ax.set_ylabel(y_col)
	ax.legend()
	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / 'data' / 'extreme.csv'
    data = pd.read_csv(data_path)
    plot_scatter_with_ellipse(data, elip_std=3.0, inlier_color=INLIER_COLOR, outlier_color=OUTLIER_COLOR, elip_ring_color=ELIP_RING_COLOR)

