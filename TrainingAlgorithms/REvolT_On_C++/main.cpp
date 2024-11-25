#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <random>

namespace HParams {
	// Optimizer params
	const double learning_rate = 3e-2;
	const double beta_1 = 0.91;
	const double beta_2 = 0.922;
	const double weight_decay = 5e-6; // Set to 0.0 for Adam-like optimization
	const double epsilon = 1e-7;
	// Model params
	const size_t in_feat = 20;
	const size_t hidden_feat = 16;
	const size_t out_feat = 1;
	// Training params
	const size_t samples = 10000;
	const size_t batch_size = 1000;
	const size_t epochs = 10;
	// Tree params
	const size_t max_depth = 8;
	const size_t tree_epochs = 2;
}

using namespace HParams;
using namespace std;

template<size_t row, size_t col>
class Matrix {
private:
	vector<vector<double>> var;
public:
	Matrix() : var(row, vector<double>(col, 0.0)) {}

	vector<double>& operator[](size_t idx) { return var[idx]; }

	template<size_t col_x>
	Matrix<row, col_x> dot(Matrix<col, col_x> x) {
		Matrix<row, col_x> result;
		for (size_t i = 0; i < row; ++i)
			for (size_t j = 0; j < col_x; ++j)
				for (size_t k = 0; k < col; ++k)
					result[i][j] += var[i][k] * x[k][j];
		return result;
	}

	Matrix<col, row> transpose() {
		Matrix<col, row> result;
		for (size_t i = 0; i < col; ++i)
			for (size_t j = 0; j < row; ++j)
				result[i][j] = var[j][i];
		return result;
	}

	vector<double> flatten() {
		vector<double> result(row*col, 0.0);
		for (size_t i = 0; i < row; ++i)
			for (size_t j = 0; j < col; ++j)
				result[i + j] = var[i][j];
		return result;
	}

	double sum() {
		double result = 0.0;
		for (size_t i = 0; i < row; ++i)
			for (size_t j = 0; j < col; ++j)
				result += var[i][j];
		return result;
	}

	Matrix<row, col> operator+= (Matrix<row, col> x) {
		for (size_t i = 0; i < row; ++i)
			for (size_t j = 0; j < col; ++j)
				var[i][j] += x[i][j];
	}

	Matrix<row, col> operator+ (Matrix<row, col> x) {
		Matrix<row, col> result;
		for (size_t i = 0; i < row; ++i)
			for (size_t j = 0; j < col; ++j)
				result[i][j] = var[i][j] + x[i][j];
		return result;
	}

	Matrix<row, col> operator- (Matrix<row, col> x) {
		Matrix<row, col> result;
		for (size_t i = 0; i < row; ++i)
			for (size_t j = 0; j < col; ++j)
				result[i][j] = var[i][j] - x[i][j];
		return result;
	}

	Matrix<row, col> operator* (Matrix<row, col> x) {
		Matrix<row, col> result;
		for (size_t i = 0; i < row; ++i)
			for (size_t j = 0; j < col; ++j)
				result[i][j] = var[i][j] * x[i][j];
		return result;
	}

	Matrix<row, col> operator/ (Matrix<row, col> x) {
		Matrix<row, col> result;
		for (size_t i = 0; i < row; ++i)
			for (size_t j = 0; j < col; ++j)
				result[i][j] = var[i][j] / x[i][j];
		return result;
	}

	Matrix<row, col> operator+ (vector<double> x) {
		Matrix<row, col> result;
		for (size_t i = 0; i < row; ++i)
			for (size_t j = 0; j < col; ++j)
				result[i][j] = var[i][j] + x[j];
		return result;
	}

	Matrix<row, col> operator+ (double x) {
		Matrix<row, col> result;
		for (size_t i = 0; i < row; ++i)
			for (size_t j = 0; j < col; ++j)
				result[i][j] = var[i][j] + x;
		return result;
	}

	Matrix<row, col> operator- (double x) {
		Matrix<row, col> result;
		for (size_t i = 0; i < row; ++i)
			for (size_t j = 0; j < col; ++j)
				result[i][j] = var[i][j] - x;
		return result;
	}

	Matrix<row, col> operator* (double x) {
		Matrix<row, col> result;
		for (size_t i = 0; i < row; ++i)
			for (size_t j = 0; j < col; ++j)
				result[i][j] = var[i][j] * x;
		return result;
	}

	Matrix<row, col> operator/ (double x) {
		Matrix<row, col> result;
		for (size_t i = 0; i < row; ++i)
			for (size_t j = 0; j < col; ++j)
				result[i][j] = var[i][j] / x;
		return result;
	}

	Matrix<row, col> operator^ (double x) {
		Matrix<row, col> result;
		for (size_t i = 0; i < row; ++i)
			for (size_t j = 0; j < col; ++j)
				result[i][j] = pow(var[i][j], x);
		return result;
	}
};

template<size_t i_f, size_t o_f>
class Neurons {
private:
	double uniform(double min = 0, double max = 1, long bound = 1e5L) { return min + (max - min) * (rand() % bound) / bound; }

	Matrix<i_f, o_f> init() {
		Matrix<i_f, o_f> weights;
		for (size_t i = 0; i < i_f; ++i)
			for (size_t j = 0; j < o_f; ++j)
				weights[i][j] = uniform();
		return weights;
	}

	Matrix<i_f, o_f> weights = init();
	Matrix<o_f, 1> biases;
public:
	void update(Matrix<i_f, o_f> w, Matrix<o_f, 1> b) {
		weights = weights - w;
		biases = biases - b;
	}

	template<size_t samples>
	Matrix<samples, o_f> forward(Matrix<samples, i_f> x) { return x.dot<o_f>(weights) + biases.flatten(); }

	Matrix<i_f, o_f> get_weights() { return weights; }

	Matrix<o_f, 1> get_biases() { return biases; }
};

class MeanSquaredError {
public:
	template<size_t samples, size_t o_f>
	double mse(Matrix<samples, o_f> target, Matrix<samples, o_f> predict) { return ((target - predict) ^ 2.0).sum() / samples; }

	template<size_t samples, size_t i_f, size_t o_f>
	Matrix<i_f, o_f> grad_w(Matrix<samples, o_f> error, Matrix<samples, i_f> input) { return input.transpose().dot<o_f>(error) * (2.0 / samples); }

	template<size_t samples, size_t o_f>
	double grad_b(Matrix<samples, o_f> error) { return error.sum() * (2.0 / samples); }
};

template<size_t i_f, size_t h_f, size_t o_f>
class AdamW {
private:
	Matrix<i_f, h_f> moments_h_w;
	Matrix<h_f, 1> moments_h_b;

	Matrix<h_f, o_f> moments_o_w;
	Matrix<o_f, 1> moments_o_b;

	Matrix<i_f, h_f> velocities_h_w;
	Matrix<h_f, 1> velocities_h_b;

	Matrix<h_f, o_f> velocities_o_w;
	Matrix<o_f, 1> velocities_o_b;
public:
	Matrix<i_f, h_f> optimize_h_w(Matrix<i_f, h_f> grad, Matrix<i_f, h_f> w, double t) {
		moments_h_w = (moments_h_w * beta_1) + (grad * (1.0 - beta_1));
		velocities_h_w = (velocities_h_w * beta_2) + ((grad ^ 2.0) * (1.0 - beta_2));
		Matrix<i_f, h_f> m_hat = moments_h_w / (1.0 - pow(beta_1, t));
		Matrix<i_f, h_f> v_hat = velocities_h_w / (1.0 - pow(beta_2, t));
		return (m_hat * learning_rate) / ((v_hat ^ 0.5) + epsilon) + (w * weight_decay);
	}

	Matrix<h_f, 1> optimize_h_b(double grad, Matrix<h_f, 1> b, double t) {
		moments_h_b = (moments_h_b * beta_1) + (grad * (1.0 - beta_1));
		velocities_h_b = (velocities_h_b * beta_2) + (pow(grad, 2.0) * (1.0 - beta_2));
		Matrix<h_f, 1> m_hat = moments_h_b / (1.0 - pow(beta_1, t));
		Matrix<h_f, 1> v_hat = velocities_h_b / (1.0 - pow(beta_2, t));
		return (m_hat * learning_rate) / ((v_hat ^ 0.5) + epsilon) + (b * weight_decay);
	}

	Matrix<h_f, o_f> optimize_o_w(Matrix<h_f, o_f> grad, Matrix<h_f, o_f> w, double t) {
		moments_o_w = (moments_o_w * beta_1) + (grad * (1.0 - beta_1));
		velocities_o_w = (velocities_o_w * beta_2) + ((grad ^ 2.0) * (1.0 - beta_2));
		Matrix<h_f, o_f> m_hat = moments_o_w / (1.0 - pow(beta_1, t));
		Matrix<h_f, o_f> v_hat = velocities_o_w / (1.0 - pow(beta_2, t));
		return (m_hat * learning_rate) / ((v_hat ^ 0.5) + epsilon) + (w * weight_decay);
	}

	Matrix<o_f, 1> optimize_o_b(double grad, Matrix<o_f, 1> b, double t) {
		moments_o_b = (moments_o_b * beta_1) + (grad * (1.0 - beta_1));
		velocities_o_b = (velocities_o_b * beta_2) + (pow(grad, 2.0) * (1.0 - beta_2));
		Matrix<o_f, 1> m_hat = moments_o_b / (1.0 - pow(beta_1, t));
		Matrix<o_f, 1> v_hat = velocities_o_b / (1.0 - pow(beta_2, t));
		return (m_hat * learning_rate) / ((v_hat ^ 0.5) + epsilon) + (b * weight_decay);
	}
};

template<size_t i_f, size_t h_f, size_t o_f>
class Model {
private:
	Neurons<i_f, h_f> hidden;
	Neurons<h_f, o_f> output;
	MeanSquaredError crit;
	AdamW<i_f, h_f, o_f> optim;

	template<size_t samples, size_t batch_size, size_t f>
	Matrix<batch_size, f> select_batch(Matrix<samples, f> a, size_t b) {
		Matrix<batch_size, f> batch;
		for (size_t i = b; i < batch_size; ++i)
			for (size_t j = 0; j < f; ++j)
				batch[i][j] = a[i][j];
		return batch;
	}

public:
	template<size_t samples>
	Matrix<samples, o_f> forward(Matrix<samples, i_f> x) { return output.forward<samples>(hidden.forward<samples>(x)); }

	template<size_t samples, size_t batch_size>
	void train(Matrix<samples, i_f> x, Matrix<samples, o_f> y, size_t epochs, bool verbose = true) {
		for (double epoch = 0.0; epoch < epochs; ++epoch) {
			for (size_t batch = 0; batch < samples; batch += batch_size) {
				Matrix<batch_size, i_f> xbatch = select_batch<samples, batch_size, i_f>(x, batch);
				Matrix<batch_size, o_f> ybatch = select_batch<samples, batch_size, o_f>(y, batch);

				Matrix<batch_size, h_f> h = hidden.forward<batch_size>(xbatch);
				Matrix<batch_size, o_f> o = output.forward<batch_size>(h);

				Matrix<batch_size, o_f> e_o = o - ybatch;
				Matrix<h_f, o_f> grad_o_w = crit.grad_w<batch_size, h_f, o_f>(e_o, h);
				double grad_o_b_ = crit.grad_b<batch_size, o_f>(e_o);
				grad_o_w = optim.optimize_o_w(grad_o_w, output.get_weights(), epoch + 1.0);
				Matrix<o_f, 1> grad_o_b = optim.optimize_o_b(grad_o_b_, output.get_biases(), epoch + 1.0);
				output.update(grad_o_w, grad_o_b);

				Matrix<batch_size, h_f> e_h = e_o.dot<h_f>(output.get_weights().transpose());
				Matrix<i_f, h_f> grad_h_w = crit.grad_w<batch_size, i_f, h_f>(e_h, xbatch);
				double grad_h_b_ = crit.grad_b<batch_size, h_f>(e_h);
				grad_h_w = optim.optimize_h_w(grad_h_w, hidden.get_weights(), epoch + 1.0);
				Matrix<h_f, 1> grad_h_b = optim.optimize_h_b(grad_h_b_, hidden.get_biases(), epoch + 1.0);
				hidden.update(grad_h_w, grad_h_b);
			}
			Matrix<samples, o_f> output = forward<samples>(x);
			double loss = crit.mse<samples, o_f>(y, output);
			if (verbose) {
				cout << "EPOCH: " << epoch << " LOSS: " << loss << "\n";
			}
		}
	}

	void copy(Model<i_f, h_f, o_f>& model) {
		hidden = model.hidden;
		output = model.output;
		optim = model.optim;
	}
};

template<size_t i_f, size_t h_f, size_t o_f, size_t samples, size_t batch_size>
class REvolT {
private:
	Model<i_f, h_f, o_f> model;
	MeanSquaredError crit;
	vector<Model<i_f, h_f, o_f>> models;
	vector<double> losses;

	template<size_t samples>
	tuple<Matrix<samples / 2, i_f>, Matrix<samples / 2, i_f>, Matrix<samples / 2, o_f>, Matrix<samples / 2, o_f>> random_split(Matrix<samples, i_f> x, Matrix<samples, o_f> y) {
		vector<size_t> idx;
		for (size_t i = 0; i < samples; ++i) { idx.push_back(i); }
		random_device rd;
		mt19937 g(rd());
		shuffle(idx.begin(), idx.end(), g);
		Matrix<samples / 2, i_f> left_x;
		Matrix<samples / 2, i_f> right_x;
		Matrix<samples / 2, o_f> left_y;
		Matrix<samples / 2, o_f> right_y;
		for (size_t i = 0; i < samples / 2; ++i) 
			for (size_t j = 0; j < i_f; ++j) 
				left_x[i][j] = x[idx[i]][j];
		for (size_t i = samples / 2; i < samples; ++i) 
			for (size_t j = 0; j < i_f; ++j) 
				right_x[i - samples / 2][j] = x[idx[i]][j];
		for (size_t i = 0; i < samples / 2; ++i) 
			for (size_t j = 0; j < o_f; ++j) 
				left_y[i][j] = y[idx[i]][j];
		for (size_t i = samples / 2; i < samples; ++i) 
			for (size_t j = 0; j < o_f; ++j) 
				right_y[i - samples / 2][j] = y[idx[i]][j];
		return make_tuple(left_x, right_x, left_y, right_y);
	}
public:
	REvolT(Model<i_f, h_f, o_f> mdl) { model = mdl; }
	void make_tree(Model<i_f, h_f, o_f> model, double loss, size_t max_depth, Matrix<samples, i_f> x, Matrix<samples, o_f> y, size_t depth = 0, bool end = false) {
		if (end) {
			models.push_back(model);
			losses.push_back(loss);
		}
		else if (depth > max_depth) { return; }
		else {
			Model<i_f, h_f, o_f> left;
			left.copy(model);
			Model<i_f, h_f, o_f> right;
			right.copy(model);
			tuple<Matrix<samples / 2, i_f>, Matrix<samples / 2, i_f>, Matrix<samples / 2, o_f>, Matrix<samples / 2, o_f>> split = random_split<samples>(x, y);
			Matrix<samples / 2, i_f> left_x = { get<0>(split) };
			Matrix<samples / 2, i_f> right_x = { get<1>(split) };
			Matrix<samples / 2, o_f> left_y = { get<2>(split) };
			Matrix<samples / 2, o_f> right_y = { get<3>(split) };
			left.train<samples / 2, batch_size>(left_x, left_y, 1, false);
			right.train<samples / 2, batch_size>(right_x, right_y, 1, false);
			double loss_left = crit.mse<samples / 2, o_f>(left_y, left.forward<samples / 2>(left_x));
			double loss_right = crit.mse<samples / 2, o_f>(right_y, right.forward<samples / 2>(right_x));
			make_tree(left, loss_left, max_depth, x, y, depth + 1, loss > loss_left);
			make_tree(right, loss_right, max_depth, x, y, depth + 1, loss > loss_right);
		}
	}

	Model<i_f, h_f, o_f> run(Matrix<samples, i_f> x, Matrix<samples, o_f> y, size_t max_depth, bool verbose = true) {
		double loss = crit.mse<samples, o_f>(y, model.forward<samples>(x));
		make_tree(model, loss, max_depth, x, y);
		auto index = distance(begin(losses), min_element(begin(losses), end(losses)));
		if (verbose) { cout << "LOSS: " << losses[index] << "\tLEAFS: " << losses.size() << "\n"; }
		return models[index];
		models.clear();
		losses.clear();
	}
};

template<size_t row, size_t col>
pair<Matrix<row, col - 1>, Matrix<row, 1>> read_csv(string path) {
	ifstream file(path);
	Matrix<row, col - 1> result_x;
	Matrix<row, 1> result_y;
	for (size_t i = 0; i < row; ++i) {
		string line;
		getline(file, line);
		stringstream s(line);
		for (size_t j = 0; j < col - 1; ++j) {
			string cell;
			getline(s, cell, ',');
			result_x[i][j] = stod(cell);
		}
		string cell;
		getline(s, cell, ',');
		result_y[i][0] = stod(cell);
	}
	file.close();
	return make_pair(result_x, result_y);
}

int main() {
	srand(time(NULL));
	pair<Matrix<samples, in_feat>, Matrix<samples, out_feat>> data = read_csv<samples, in_feat+out_feat>("classification.csv");
	Matrix<samples, in_feat> x = data.first;
	Matrix<samples, out_feat> y = data.second;
	Model<in_feat, hidden_feat, out_feat> mdl_0;
	cout << "Random Evolutionary Training:\n";
	for (size_t epoch = 0; epoch < tree_epochs; ++epoch) {
		REvolT<in_feat, hidden_feat, out_feat, samples, batch_size> evol(mdl_0);
		mdl_0 = evol.run(x, y, max_depth);
	}
	Model<in_feat, hidden_feat, out_feat> mdl_1;
	cout << "Classic Training:\n";
	mdl_1.train<samples, batch_size>(x, y, epochs);
}
