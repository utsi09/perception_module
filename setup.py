from setuptools import find_packages, setup

package_name = 'fastsam_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
      'setuptools',
      'ultralytics',
      'opencv-python',
      'numpy',
    ],
    zip_safe=True,
    maintainer='taewook',
    maintainer_email='utsi09@g.skku.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fastsam_node = fastsam_ros.fastsam_node:main',
        ],
    },
)
