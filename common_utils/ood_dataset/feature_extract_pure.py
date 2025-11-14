import pickle
import torch.nn.functional as F
import os
from tqdm import tqdm
# 指定GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import alexnet
import torch
import torch.nn as nn
from torchvision import models

class feature_resnet18(nn.Module):
    def __init__(self, is_frozen=False):
        super(feature_resnet18, self).__init__()
        self.net = models.resnet18(pretrained=True).cuda()
        # 冻结参数，防止微调
        if is_frozen:
            for param in self.net.parameters():
                param.requires_grad = False
            print(f"参数冻结！！")
        # for param in self.net.parameters():
        #     param.requires_grad = False
    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        return x

class feature_resnet50(nn.Module):
    def __init__(self, is_frozen=False):
        super(feature_resnet50, self).__init__()
        self.net = models.resnet50(pretrained=True)
        # 冻结参数，防止微调
        if is_frozen:
            for param in self.net.parameters():
                param.requires_grad = False
            print(f"参数冻结！！")
    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        return x



NET_OUT_DIM = {'alexnet': 256, 'resnet18': 512, 'resnet50': 2048}


def build_feature_extractor(backbone, is_frozen=False):
    if backbone == "alexnet":
        print("Feature extractor: AlexNet")
        return alexnet(pretrained=True).features
    elif backbone == "resnet18":
        print("Feature extractor: ResNet-18")
        return feature_resnet18(is_frozen)
    elif backbone == "resnet50":
        print("Feature extractor: ResNet-50")
        return feature_resnet50(is_frozen)
    else:
        raise NotImplementedError

def visualize_features(X_data, labels, classes, dataset_name, method='tsne', save_path=None, is_test=None):
    """
    使用指定方法可视化特征分布

    Args:
        X_data: 特征数据 (N, D)
        labels: 类别标签列表
        classes: 唯一类别列表
        dataset_name: 数据集名称
        method: 降维方法 ('tsne', 'umap', 'pca')
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import umap
    import matplotlib.pyplot as plt

    # print(f"\n使用 {method.upper()} 进行降维可视化...")

    # 降维到2D
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1,n_epochs=200, )  #  low_memory=True
        X_data = PCA(n_components=50, random_state=42).fit_transform(X_data)  # X 是高维特征
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError(f"不支持的降维方法: {method}")

    X_reduced = reducer.fit_transform(X_data)

    # 绘制散点图
    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))

    for i, cls in enumerate(classes):
        mask = np.array(labels) == cls
        plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                    c=[colors[i]], label=cls, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)

    plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Component 2', fontsize=12)
    plt.title(f'Feature Distribution - {dataset_name} {"Test" if is_test else "Train"}({method.upper()})', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()
    return X_reduced  # 返回降维结果
def load_image(path):
    if 'npy' in path[-3:]:
        img = np.load(path).astype(np.uint8)
        img = img[:, :, :3]
        return Image.fromarray(img)
    return Image.open(path).convert('RGB')

def show_emb(X_all, methods=[ 'umap'], dataset="default", save_path=None, is_test=False):
    # 合并训练集和测试集进行可视化 ['umap','pca', 'tsne']
    # 提取特征和类别标签
    features = []
    labels = []

    # 按key排序，确保顺序一致
    for key in sorted(X_all.keys()):
        val = X_all[key]
        cls = key.split('/')[-2]  # 提取类别名称
        features.append(val.flatten())  # 展平特征
        labels.append(cls)

    features = np.array(features)

    # 统计类别分布
    unique_classes, counts = np.unique(labels, return_counts=True)
    # print(f"\n类别分布:")
    # for cls, cnt in zip(unique_classes, counts):
    #     print(f"  {cls}: {cnt} 样本")

    # 使用三种方法进行可视化
    X_reduced_dict = {}
    for method in methods:
        try:
            X_reduced = visualize_features(features, labels, unique_classes, dataset, method=method, save_path=save_path, is_test=is_test)
            # 构建降维结果
            X_reduced_dict_tmp = {key: X_reduced[i] for i, key in enumerate(sorted(X_all.keys()))}
            X_reduced_dict[method] = X_reduced_dict_tmp

        except Exception as e:
            print(f"⚠️ {method.upper()} 可视化失败: {str(e)}")
    return X_reduced_dict
# 这里的X_train可以是图片本身（比较规则？并不规则，需要transform）
def get_emb(X_all, batch_size, feature_extractor):
    """高效提取特征embedding"""
    X_all_keys = list(X_all.keys())
    X_all_vals = list(X_all.values())
    X_train_dict = {}
    X_test_dict = {}
    with torch.no_grad():
        for i in tqdm(range(0, len(X_all), batch_size)):
            batch_end = min(i + batch_size, len(X_all_vals))
            X_batch = X_all_vals[i:batch_end]   # tensor
            batch_paths = X_all_keys[i:batch_end]
            batch_tensor = torch.stack(X_batch ).cuda()  # B x C x H x W
            # 直接在GPU上操作
            # x = torch.from_numpy(X_batch).float().cuda()
            x = feature_extractor(batch_tensor)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)

            # 转为 numpy 并保存
            features_np = x.cpu().numpy()
            for j, path in enumerate(batch_paths):
                if 'train' in path:
                    X_train_dict[path] = features_np[j]
                else:
                    X_test_dict[path] = features_np[j]

        return X_test_dict | X_train_dict
def get_normal(X_all, batch_size, feature_extractor, ):
    # feature_extractor.eval()
    X_all_keys = list(X_all.keys())
    X_all_vals = list(X_all.values())
    X_train_dict = {}
    X_test_dict = {}
    with torch.no_grad():
        for i in tqdm(range(0, len(X_all), batch_size)):
            batch_paths = X_all_keys[i:i + batch_size]
            batch_images = X_all_vals[i:i + batch_size]

            # 转为 tensor 并送入 GPU
            batch_tensor = torch.stack(batch_images).cuda()  # B x C x H x W

            # 提取特征（不计算梯度）

            features = feature_extractor(batch_tensor)  # shape: [B, D]，D = NET_OUT_DIM
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
            # 转为 numpy 并保存
            features_np = features.cpu().numpy()
            for j, path in enumerate(batch_paths):
                if 'train' in path:
                    X_train_dict[path] = features_np[j]
                else:
                    X_test_dict[path] = features_np[j]
    return X_test_dict | X_train_dict
signatrue = 'normal'
def get_feature():
    # 参数
    is_frozen = False  # 启动eval
    use_eval = False
    dataset_list = ['carpet']
    img_size = 224  # 假设使用标准 ImageNet 尺寸，可根据实际调整
    batch_size = 32  # 可选：支持批量推理加速

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 特征提取器（ResNet18，输出特征维度由 NET_OUT_DIM 决定）
    feature_extractor = build_feature_extractor('resnet18', is_frozen=is_frozen)
    feature_extractor = feature_extractor.to(device)
    if use_eval:  # bug原因：使用eval后AUC能有97%，不用eval只有33%
        feature_extractor.eval()
    else:
        feature_extractor.train()

    # 2. 定义与训练一致的图像变换（注意：此处应与训练时的 transform_test 一致）
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. 遍历数据集并提取特征
    for dataset in dataset_list:
        file_path = r'/data/coding/wsad/zsy/WSADBench/WSADBench/datasets/CV_OOD/' + dataset
        print(f"Processing dataset: {dataset} at {file_path}")

        # 收集所有图像路径
        image_paths = []
        # 存储路径 -> 特征的映射
        X_train_dict = {}
        X_test_dict = {}
        for root, dirs, files in os.walk(file_path):
            for file in files:
                dataset_type = root.split('/')[-2]
                if file.lower().endswith(('.jpg', '.png')) and dataset_type in ['train', 'test']:
                    # image_paths.append(os.path.join(root, file))
                    path = os.path.join(root, file)
                    img = load_image(path)
                    img_t = transform(img)  # C x H x W
                    if dataset_type == 'train':
                        X_train_dict[path] = img_t
                    else:
                        X_test_dict[path] = img_t
        X_all = X_test_dict | X_train_dict
        print(f"Found {len(X_all)} images.")


        # 注入图片
        # img_path  = fr'/data/coding/wsad/zsy/WSADBench/WSADBench/datasets/CV_by_ResNet18_OOD_pic/{dataset}_pic.pkl'
        # with open(img_path, 'rb') as f:
        #     X_train_dict_tmp = pickle.load(f)
        #     X_test_dict_tmp = pickle.load(f)
        #     X_all = X_test_dict_tmp | X_train_dict_tmp
            # image_paths = list(X_all.keys())
            # X_all =  list(X_all.values())
        # 分批处理（避免内存溢出）



        # 法1 用get_emb
        if signatrue == 'get_emb':
            X_all = get_emb(X_all, batch_size, feature_extractor)
        # 法2 直观实现
        elif signatrue == 'normal':
            X_all = get_normal(X_all, batch_size, feature_extractor)
        else:
            raise NotImplementedError


        # 可视化
        # X_all = X_train_dict | X_test_dict
        X_reduced_dict = show_emb(X_all=X_all, dataset=dataset, is_test=False)


        # 4. 保存特征（例如保存为 .npy 文件）
        X_train_dict = {}  # 拆分
        X_test_dict = {}
        for key, value in X_all.items():
            if 'train' in key:
                X_train_dict[key] = value
            else:
                X_test_dict[key] = value
        save_dir = f"/data/coding/wsad/zsy/WSADBench/myRes/emb_pure/{dataset}"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{dataset}_{signatrue}.pkl")
        # 保存
        with open(save_path, 'wb') as f:
            pickle.dump({"X_train_dict":X_train_dict, "X_test_dict":X_test_dict, "X_reduced_dict":X_reduced_dict}, f)
        print(f"Saved features to {save_path}, shape: {len(X_train_dict)} {len(X_test_dict)}")


if __name__ == "__main__":
    get_feature()