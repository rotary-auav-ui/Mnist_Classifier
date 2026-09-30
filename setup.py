# from setuptools import find_packages, setup

# package_name = 'mnist_classifier'

# setup(
#     name=package_name,
#     version='0.0.0',
#     packages=find_packages(exclude=['test']),
#     data_files=[
#         ('share/ament_index/resource_index/packages',
#             ['resource/' + package_name]),
#         ('share/' + package_name, ['package.xml']),
#     ],
#     install_requires=['setuptools'],
#     zip_safe=True,
#     maintainer='adri',
#     maintainer_email='adriyudosatria@gmail.com',
#     description='TODO: Package description',
#     license='TODO: License declaration',
#     extras_require={
#         'test': [
#             'pytest',
#         ],
#     },
#     entry_points={
#         'console_scripts': [
#             'mnist_classifier_node = mnist_classifier.classifier:main',
#         ],
#     },
# )
from setuptools import setup
import os
from glob import glob


package_name = "mnist_classifier"


setup(
    name=package_name,
    version="0.0.0",

    packages=[package_name],

    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name]
        ),
(
    "share/" + package_name + "/models",
    ["mnist_classifier/simple_cnn_mnist.pth"]
),
        (
            "share/" + package_name,
            ["package.xml"]
        ),
    ],

    install_requires=["setuptools"],

    zip_safe=True,

    maintainer="your_name",
    maintainer_email="your@email.com",

    description="ROS 2 MNIST CNN classifier",

    license="Apache-2.0",

    entry_points={
        "console_scripts": [
            "mnist_node = "
            "mnist_classifier.classifier:main",
        ],
    },
)