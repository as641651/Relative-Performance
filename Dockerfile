FROM centos:centos7.6.1810

RUN yum -y update
RUN yum -y install vim
RUN yum -y install redhat-lsb
RUN yum -y install epel-release
RUN yum -y install python36-pip
RUN yum -y install git
RUN yum -y install bzip2
RUN yum clean all

# gcc 7
RUN yum -y install centos-release-scl
RUN yum -y install devtoolset-7-gcc-c++

# CMake 3.10.1
RUN git clone https://github.com/Kitware/CMake.git
WORKDIR /CMake
RUN git checkout tags/v3.10.1
RUN . /opt/rh/devtoolset-7/enable \
	&& ./bootstrap \
	&& make \
	&& make install
WORKDIR /

# Intel MKL 2018.2
RUN yum-config-manager --add-repo https://yum.repos.intel.com/mkl/setup/intel-mkl.repo
RUN . /opt/rh/devtoolset-7/enable && yum -y --nogpgcheck install intel-mkl-2018.2-046

# julia 1.1.0
RUN git clone https://github.com/JuliaLang/julia.git
WORKDIR julia/
RUN git checkout 1c6f89f04a1ee4eba8380419a2b01426e84f52aa
RUN echo "USE_INTEL_MKL = 1" > Make.user
RUN . /opt/rh/devtoolset-7/enable && . /opt/intel/mkl/bin/mklvars.sh intel64 verbose && make

RUN pip3 install jupyter

RUN ./julia -e "using Pkg; Pkg.add(PackageSpec(url=\"https://github.com/HPAC/MatrixGenerator.jl.git\", rev=\"master\"))"
WORKDIR /

RUN echo 'source /opt/rh/devtoolset-7/enable' >> ~/.bashrc
RUN echo 'source /opt/intel/mkl/bin/mklvars.sh intel64 verbose' >> ~/.bashrc
RUN echo 'alias julia="/julia/julia"' >> ~/.bashrc

RUN pip3 install --upgrade pip
RUN pip3 install matplotlib
RUN yum -y install python3-devel
RUN pip install --upgrade tensorflow
RUN pip3 install pandas

RUN mkdir Docker
COPY . Docker/

#RUN adduser user
#USER user

WORKDIR /home/user

CMD ["bash"]
