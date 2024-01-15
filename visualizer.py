import numpy as np

class Visualizer():
    def __init__(self, visdom_port, env_name):
        self.display_id = 1
        self.win_size = 256
        if self.display_id > 0:
            import visdom
            # /opt/anaconda3/bin/python -m visdom.server -port 8097
            # 防火墙允许8097端口使用：iptables -A INPUT -p tcp --dport 8097 -j ACCEPT
            self.vis = visdom.Visdom(server="http://localhost", port=visdom_port, env=env_name, raise_exceptions=True)

    def throw_visdom_connection_error(self):
        print('\n\nCould not connect to Visdom server (https://github.com/facebookresearch/visdom) for displaying training progress.\nYou can suppress connection to Visdom using the option --display_id -1. To install visdom, run \n$ pip install visdom\n, and start the server by \n$ python -m visdom.server.\n\n')
        exit(1)

    # def plot_acc_and_loss(self, epoch, acc, loss):
    #     if not hasattr(self, 'plot_data'):
    #         self.plot_data = {'X': [], 'Y': [], 'legend':['acc', 'loss']}
    #
    #     self.plot_data['X'].append(epoch)
    #     self.plot_data['Y'].append([acc, loss])
    #     try:
    #         self.vis.line(
    #             X=np.array(self.plot_data['X']),
    #             Y=np.array(self.plot_data['Y']),
    #             opts={
    #                 'title': 'acc and loss over time',
    #                 'legend': self.plot_data['legend'],
    #                 'xlabel': 'epoch',
    #                 'ylabel': 'acc/loss'},
    #             win=self.display_id)
    #     except ConnectionError:
    #         self.throw_visdom_connection_error()

    def plot_acc(self, epoch, acc):
        if not hasattr(self, 'acc_data'):
            self.acc_data = {'X': [], 'Y': []}

        self.acc_data['X'].append(epoch)
        self.acc_data['Y'].append(acc)
        try:
            self.vis.line(
                X=np.array(self.acc_data['X']),
                Y=np.array(self.acc_data['Y']),
                opts={
                    'title': 'acc over time',
                    'xlabel': 'epoch',
                    'ylabel': 'acc',
                    'color':'red'}, # 版本太低不起作用
                win=1)
        except ConnectionError:
            self.throw_visdom_connection_error()

    def plot_loss(self, epoch, loss):
        if not hasattr(self, 'loss_data'):
            self.loss_data = {'X': [], 'Y': []}

        self.loss_data['X'].append(epoch)
        self.loss_data['Y'].append(loss)
        try:
            self.vis.line(
                X=np.array(self.loss_data['X']),
                Y=np.array(self.loss_data['Y']),
                opts={
                    'title': 'loss over time',
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=2)
        except ConnectionError:
            self.throw_visdom_connection_error()


if __name__ == '__main__':
    import visdom
    import numpy as np

    # 创建Visdom客户端
    vis = visdom.Visdom(server="http://localhost", port=8097, env='uk_10e3_model3_(1)', raise_exceptions=True)

    # 生成数据
    X = np.array([0, 1, 2, 3, 4])
    Y = np.array([0, 1, 4, 9, 16])

    # 设置线的颜色为红色
    opts = {
        'title': 'acc over time',
        'xlabel': 'epoch',
        'ylabel': 'acc'
    }

    # 绘制线图
    img1 = vis.line(X=X, Y=Y, win=1, opts=opts, name='my_test_img1')
    img2 = vis.line(X=X, Y=Y, win=2, opts=opts, name='my_test_img2')