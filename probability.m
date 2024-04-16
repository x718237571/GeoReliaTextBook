clc; clear; close all


%% ANN拟合函数
clc; clear; close all
tic

n=30;
lim=70;
x=randperm(lim,n);
y=x.^4+x.^3+9;
plot(x,y,'ro')
hold on
net = fitnet(10);
net= train(net,x,y); 

x1=0:3:lim;
g = sim(net,x1);
plot(x1,g,'b')
hold off

%% ANN-FOSM structure reliability -ZhangMing
clc; clear; close all
tic

muX = [38;54]; sigmaX = [3.8;2.7]; 
nP = 100; 
x = [normrnd(muX(1),sigmaX(1),nP,1),normrnd(muX(2),sigmaX(2),nP,1)]; 
g = x(:,1).*x(:,2)-1140;
xr = minmax(x');
net = newff(xr, [5, 1],{'tansig','purelin'});
% net = fitnet(10);
net= train(net,x',g'); 
x = muX; normX = eps; 
while abs(norm(x)-normX)/normX>1e-6 
    normX = norm(x); 
    g = sim(net,x); 
    z1 = net.iw{1,1}*x+net.b{1};  % 权重矩阵、阈值
    z2 = net.lw{2,1}*tansig(z1)+net.b{2}; 
    df1 = tansig('dn',z1); 
    df2 = purelin('dn',z2); 
    gX = (net.iw{1,1})'*(df1.*(net.lw{2,1})')*df2; 
    gs= gX.*sigmaX; alphaX = -gs/norm(gs); 
    bbeta = (g+gX'*(muX-x))/norm(gs);
    x = muX+bbeta*sigmaX.*alphaX;
end
toc

%% ANN-MC structure reliability -ZhangMing
clc; clear; close all
tic

muX=[0.36;0.18;20];cvX=[0.1;0.1;0.25];
sigmaX=cvX.*muX;
sLn=sqrt(log(1+(sigmaX(1:2)./muX(1:2)).^2));
mLn=log(muX(1:2))-sLn.^2/2;
aEv=sqrt(6)*sigmaX(3)/pi;
uEv=-psi(1)*aEv-muX(3);
nP=150;
x=[lognrnd(mLn(1),sLn(1),nP,1),lognrnd(mLn(2),sLn(2),nP,1),-evrnd(uEv,aEv,nP,1)];
e=2e6;
i1=x(:,1).^2/12; i2=x(:,2).^2/6; k=i2./i1;
g=0.01-(48*k+32)./(18*k+3).*x(:,3)/e./i1;
net=newff(x',g',20);
[net,tr]=train(net,x',g');
nS=1e7; nS1=1e5; ig=ones(nS1,1); nF=0;
for k=1:nS/nS1
    x=[lognrnd(mLn(1),sLn(1),nS1,1),lognrnd(mLn(2),sLn(2),nS1,1),-evrnd(uEv,aEv,nS1,1)];
    g1=sim(net,x'); % 神经网络仿真
    nF=nF+sum(ig(g1<0));
end
pF=nF/nS
toc

%% Reliability solving by iterations of vector projection method
% clc; clear; close all
% tic
% 
% muX = [2;2]; sigmaX =[0.3;0.3]; correlationX = [1 0;0 1]; n = length(muX);
% alpha = 2; % 初值取 2
% Amatrix = CCD(muX, sigmaX, alpha); % 设计点
% Zresponse = Gfunction ( Amatrix (:,2: n + 1)); % 求响应
% Lambda= inv(Amatrix(:,1:n+1)'* Amatrix(:,1:n+1)) * Amatrix(:,1:n+1)'* Zresponse; % 最小二乘法，其中A(:,1:n+1)是一次多项式的，求3个值
% [beta, xstar] = ReliabilityFor1stPolynomial(muX, sigmaX, correlationX, Lambda); % 求初始1次响应面的 设计验算点 和 可靠度指标
% omega = 0.5; epsq = 0; tempbeta = 0; count = 1; % omega为加权因子
% 
% % while abs(beta-tempbeta) > 1e-6
% while count<50
%     tempbeta = beta; centerX = xstar; Oldlambda = Lambda;
%     normalvector = - Lambda(2:n+1)'./norm(Lambda(2:n+1)'); % 归一化
%     Amatrix1storder = VectorProjectionDesign (centerX, sigmaX, alpha, omega, normalvector, epsq); % 设计点的更新，命名可能想说store
%     Zresponse = Gfunction ( Amatrix1storder(:,2:n+1)) ;  % Amatrix1storder(:,2:n+ 1)是设计点
%     Lambda = inv(Amatrix1storder'*Amatrix1storder) * Amatrix1storder'*Zresponse; % 最小二乘法
%     [beta,xstar] = ReliabilityFor1stPolynomial(muX, sigmaX, correlationX, Lambda); % 新的中心设计点
%     count = count+1; alpha = alpha^0.5;
% end
% 
% x1mesh = (0:0.01:4)'; % 画图区间
% x2RealZeroPlane = ((x1mesh(:,1)-1).^3 +x1mesh(:,1)+2)./(x1mesh(:,1)); % Z=0移项求x2
% x2FitZeroPlane = -(Lambda(1)+Lambda(2)*x1mesh(:,1))/Lambda(3); % Z=0移项求x2
% plot(x1mesh, x2RealZeroPlane,'-r','linewidth',2) % 真实极限状态面
% hold on; grid on; axis square; axis([0 4 1 5]);
% plot(x1mesh, x2FitZeroPlane, '-.b ','linewidth',2) % 投影法取样的响应面
% x2OldZeroPlane = -(Oldlambda(1)+Oldlambda(2)*x1mesh(:,1) )/Oldlambda(3); % Z=0移项求x2
% plot(x1mesh, x2OldZeroPlane, ':m ','linewidth',2) % 上次迭代响应面
% plot(Amatrix1storder(:,2),Amatrix1storder(:,3), 'o') % 设计点
% hold off
% toc
% 
% % 投影向量法生成一系列新的设计点
% function [Amatrix1storder] = VectorProjectionDesign(centerX, sigmaX, alpha,omega, normalvector, epsq)
% n = length(sigmaX);
% E = eye(n); % eye(n): diagonal unit matrix for axis unit vector of variable X
% Eprojection = dot(eye(n), repmat(normalvector,n,1))'*normalvector; % Projection of axis unit vector
% T = eye(n) - Eprojection; % substraction of vector
% for j = 1:n
%     T(j,:) =T(j,:)./norm(T(j,:)); % 正则化
% end
% q = omega*(T+epsq*repmat(normalvector,n,1))+(1-omega)*E; % 近旁投影选取设计点
% for j= 1:n
%     q(j,:) = q(j,:)./norm(q(j,:)); % 正则化
% end
% sigmaq = (repmat(alpha*sigmaX,1,n)).*q ; % 求取样点与设计验算点距离
% column1 = ones(2*n+1,1); 
% row1=[centerX'];
% c1 = repmat(centerX',n,1); 
% row2n = [c1-sigmaq]; 
% row3n = [c1+sigmaq];
% Amatrix1storder = [column1, [row1; row2n; row3n]];
% end
% 
% function [beta,xstar] = ReliabilityFor1stPolynomial(muX, sigmaX, correlationX,lambda)
% covarianceX = diag(sigmaX)*correlationX*diag(sigmaX) ;
% [orthogonalYtoX,diagsigmaY2] = eig(covarianceX);
% muY = orthogonalYtoX'*muX; sigmaY = sqrt(diag(diagsigmaY2));
% n = length(muX); x=muX; y=muY; normX = eps;
% while abs(norm(x)-normX)/normX > 1e-6
% normX = norm( x) ;
% g = lambda'* [1;x];
% deltagX = lambda(2:n+1); % 因为只是一次的求导，好像就不用delta_g函数了
% deltagY = orthogonalYtoX'*deltagX;
% gs = deltagY.*sigmaY;
% alphaY = -gs/norm(gs);
% beta = (g+deltagY'*(muY-y))/norm(gs);
% y = muY + beta*sigmaY.* alphaY;
% x = orthogonalYtoX * y;
% end
% xstar = x;
% end
% 
% function g=Gfunction(x)
% g=(x(:,1)-1).^3-x(:,2).*(x(:,1))+(x(:,1))+2;
% end
% 
% function [A]=CCD(mu_x,sigma_x,alpha)
% x=mu_x;
% n=length(mu_x);
% diag_sigma=diag(alpha*sigma_x);
% column1=ones(2*n+1,1);
% b1=x'; % x转置后为行向量
% b2=(x.*x)';
% row1=[b1 b2];
% c1=repmat(x',n,1); % 把x'重复n行
% c2=c1-diag_sigma;
% c3=c2.*c2;
% d2=c1+diag_sigma;
% d3=d2.*d2;
% row2n=[c2,c3];
% row3n=[d2,d3];
% A=[column1,[row1;row2n;row3n]];
% end

%% RSM迭代求解 6-3
% clc; clear; close all
% tic
% 
% muX = [2;2]; sigmaX = [0.3;0.3];
% correlationX = [1,0;0,1]; n = length(muX);
% beta = 1; tempbeta = 0;
% alpha = 2; 
% CCDcenterX = muX; count = 1; 
% GmuX = Gfunction(muX'); 
% while abs(beta-tempbeta) > 1e-6
% tempbeta = beta;
% Amatrix = CCD(CCDcenterX,sigmaX,alpha);
% Zresponse = Gfunction( Amatrix(:,2:n+1)); 
% Lambda = Amatrix\Zresponse; 
% [beta,xstar] = reliability (muX,sigmaX,correlationX);
% count = count + 1; alpha = alpha.^0.5;
% Gxstar = Gfunction(xstar');
% CCDcenterX = muX - GmuX * (xstar-muX)/(Gxstar-GmuX);
% end
% 
% x1mesh = (0:0.01:4)';
% x2RealZeroPlane =((x1mesh(:,1)-1).^3 +x1mesh(:,1)+2)./(x1mesh(:,1)); % Z=0移项求x2
% x2FitZeroPlane = ZeroPlaneSolver2ndPolynomial(x1mesh, Lambda); % Z=0迭代求x2
% plot ( x1mesh, x2RealZeroPlane,'-r ','LineWidth', 2)  % 真实极限状态面
% hold on; grid on; axis square; axis([0 4 1 5]) ;
% plot( x1mesh, x2FitZeroPlane,'-.b ','LineWidth',2)  % 响应面
% plot(Amatrix(:,2) ,Amatrix(:,3) ,'o') %draw design points
% hold off
% 
% function g=Gfunction(x)
% g=(x(:,1)-1).^3-x(:,2).*(x(:,1))+(x(:,1))+2;
% end
% 
% function delta_g=delta_Gfunc(x)
% delta_g=[3*(x(:,1)-1).^2-x(:,2)+1;-x(:,1)]; % 两个偏导部分组成
% end
% 
% function [A]=CCD(mu_x,sigma_x,alpha)
% x=mu_x;
% n=length(mu_x);
% diag_sigma=diag(alpha*sigma_x);
% column1=ones(2*n+1,1);
% b1=x'; % x转置后为行向量
% b2=(x.*x)';
% row1=[b1 b2];
% c1=repmat(x',n,1); % 把x'重复n行
% c2=c1-diag_sigma;
% c3=c2.*c2;
% d2=c1+diag_sigma;
% d3=d2.*d2;
% row2n=[c2,c3];
% row3n=[d2,d3];
% A=[column1,[row1;row2n;row3n]];
% end
% 
% function [beta,x1]=reliability(mu_x,sigma_x,cor_x)
% cov_x=diag(sigma_x)*cor_x*diag(sigma_x);
% [V,D]=eig(cov_x);
% mu_y=V'*mu_x; % 正交化
% sigma_y=sqrt(diag(D));
% x=mu_x; y=mu_y;
% normx=eps;
% while abs(norm(x)-normx)/normx>1e-5
%     normx=norm(x);
%     g=Gfunction(x'); % 二次响应面函数
%     delta_gx=delta_Gfunc(x');
%     delta_gy=V'*delta_gx;
%     gs=delta_gy.*sigma_y;
%     alpha_y=-gs/norm(gs);
%     beta=(g+delta_gy'*(mu_y-y))/norm(gs);
%     y=mu_y+beta*sigma_y.*alpha_y;
%     x=V*y;
% end
% x1=x;
% end
% 
% % 因为求z=0时，x2式中有x2的二次项所以不能简单通过移项得到，需要迭代求解
% function [x2FitZeroPlane] = ZeroPlaneSolver2ndPolynomial(x1mesh,lambda)
% n = length(x1mesh);
% for j = 1:1:n
%     x2 = 1;
%     normx2 = eps;
%     while abs(norm(x2)-normx2)/normx2>1e-5
%         x2 = lambda'* [1;x1mesh(j,1);0;x1mesh(j,1).*x1mesh(j,1);x2.*x2];
%         x2 = x2*(-1)/lambda(3);
%         normx2=norm(x2);
%     end
%     x2FitZeroPlane(j,1)= x2;
% end
% end

%% 研究alpha对近似结果的影响 6-2
clc; clear; close all

mu_x=[2;2]; 
sigma_x=[0.3;0.3];
cor_x=[1 0;
       0 1];
n = length(mu_x);
alpha = 6;
A = CCD(mu_x,sigma_x,alpha);
Z = Gfunc(A(:,2:n+1)); 
lambda=A\Z;
[beta,x1] = reliability(mu_x, sigma_x,cor_x);
x1mesh = (0.01:0.01:4)';
x2RealZeroPlane = (( x1mesh(:, 1) -1).^3 + x1mesh(:,1) +2)./(x1mesh(:,1)); % Z=0移项求x2
x2FitZeroPlane = ZeroPlaneSolver2ndPolynomial(x1mesh,lambda); % Z=0迭代求x2
plot( x1mesh, x2RealZeroPlane,'-r','linewidth',2) % 真实极限状态面
hold on; grid on; axis square; axis([0 4 0 4]);
plot(x1mesh, x2FitZeroPlane,'-.b','linewidth',2) % 响应面
plot(A(:,2),A(:,3),'o')
hold off
xlabel('X1');ylabel('X2');legend('Real LSS','RSM','DoE','location','southeast')
% 因为求z=0时，x2式中有x2的二次项所以不能简单通过移项得到，需要迭代求解
function [x2FitZeroPlane] = ZeroPlaneSolver2ndPolynomial(x1mesh,lambda) 
n = length(x1mesh);
for j = 1:n
    x2 = 1;
    normx2 = eps;
    while abs(norm(x2)-normx2)/normx2>1e-5
        x2 = lambda'* [1;x1mesh(j,1);0;x1mesh(j,1).*x1mesh(j,1);x2.*x2];
        x2 = x2*(-1)/lambda(3);
        normx2=norm(x2);
    end
    x2FitZeroPlane(j,1)= x2;
end
end

function g=Gfunc(x)
g=(x(:,1)-1).^3-x(:,2).*(x(:,1))+(x(:,1))+2;
end

function delta_g=delta_Gfunc(x)
delta_g=[3*(x(:,1)-1).^2-x(:,2)+1; -x(:,1)]; % 两个偏导部分组成
end

function [beta,x1]=reliability(mu_x,sigma_x,cor_x)
cov_x=diag(sigma_x)*cor_x*diag(sigma_x);
[V,D]=eig(cov_x);
mu_y=V'*mu_x; % 正交化
sigma_y=sqrt(diag(D));
x=mu_x; y=mu_y;
normx=eps;
while abs(norm(x)-normx)/normx>1e-5
    normx=norm(x);
    g=Gfunc(x'); % 二次响应面函数
    delta_gx=delta_Gfunc(x');
    delta_gy=V'*delta_gx;
    gs=delta_gy.*sigma_y;
    alpha_y=-gs/norm(gs);
    beta=(g+delta_gy'*(mu_y-y))/norm(gs);
    y=mu_y+beta*sigma_y.*alpha_y;
    x=V*y;
end
x1=x;
end

function [A]=CCD(mu_x,sigma_x,alpha)
x=mu_x;
n=length(mu_x);
diag_sigma=diag(alpha*sigma_x);
column1=ones(2*n+1,1);
b1=x'; % x转置后为行向量
b2=(x.*x)';
row1=[b1 b2];
c1=repmat(x',n,1); % 把x'重复n行
c2=c1-diag_sigma;
c3=c2.*c2;
d2=c1+diag_sigma;
d3=d2.*d2;
row2n=[c2,c3];
row3n=[d2,d3];
A=[column1,[row1;row2n;row3n]];
end

%% Response Surface Method 6-1
% clc; clear; close all
% 
% alpha=1.414;
% mu_x=[10;0.36];
% sigma_x=[3;0.06];
% A=CCD(mu_x,sigma_x,alpha)
% z=[0.165;-0.059;0.001;0.365;0.307];
% lambda=A\z
% cor_x=[1 0;
%        0 1];
% [beta,x1]=rpolynomial(mu_x,sigma_x,cor_x,lambda)
% p=1-normcdf(beta)
% 
% 
% function [A]=CCD(mu_x,sigma_x,alpha)
% x=mu_x;
% n=length(mu_x);
% diag_sigma=diag(alpha*sigma_x);
% column1=ones(2*n+1,1);
% b1=x'; % x转置后为行向量
% b2=(x.*x)';
% row1=[b1 b2];
% c1=repmat(x',n,1); % 把x'重复n行
% c2=c1-diag_sigma;
% c3=c2.*c2;
% d2=c1+diag_sigma;
% d3=d2.*d2;
% row2n=[c2,c3];
% row3n=[d2,d3];
% A=[column1,[row1;row2n;row3n]];
% end
% 
% function [beta,x1]=rpolynomial(mu_x,sigma_x,cor_x,lambda)
% cov_x=diag(sigma_x)*cor_x*diag(sigma_x);
% [V,D]=eig(cov_x);
% mu_y=V'*mu_x; % 正交化
% sigma_y=sqrt(diag(D));
% x=mu_x; y=mu_y;
% normx=eps;
% n=length(mu_x);
% while abs(norm(x)-normx)/normx>1e-6
%     normx=norm(x);
%     g=lambda'*[1;x;x.*x]; % 二次响应面函数
%     delta_gx=lambda(2:n+1)+2*lambda(n+2:2*n+1).*x;
%     delta_gy=V'*delta_gx;
%     gs=delta_gy.*sigma_y;
%     alpha_y=-gs/norm(gs);
%     beta=(g+delta_gy'*(mu_y-y))/norm(gs);
%     y=mu_y+beta*sigma_y.*alpha_y;
%     x=V*y;
% end
% x1=x;
% end

%% Latin Hypercube Sampling 内置LHS抽样函数

% n=50000;
% mu_x=[20;30];
% sigma=[5;6];
% C=[25 -15;
%     -15 36];
% q=500;
% z=lhsnorm(mu_x,C,n)'; % 内置LHS抽样函数
% 
% for i=1:n
%     qu(i)=Hansen(z(:,i)');
%     Fs(i,1)=qu(i)/q;
%     if Fs(i,1)<1
%         I(i,1)=1;
%     else
%         I(i,1)=0;
%     end
% end
% pf_m=mean(I)
% pf_std=std(I)/sqrt(n);
% pf_cov=pf_std/pf_m
% toc
% 
% function qu=Hansen(x) %汉森公式
% B=2;
% Df=0.5;
% gs=20; %重度
% c=x(1); %粘聚力
% fai=x(2)*pi/180; %标准差6度角度改为弧度
% Nq=(tan(pi/4+fai/2))^2*exp(pi*tan(fai));
% Ng=1.8*(Nq-1)*exp(pi*tan(fai));
% Nc=(Nq-1)*cot(fai);
% qu=0.5*gs*B*Ng+c*Nc+gs*Df*Nq; %承载力
% end

%% Latin Hypercube Sampling

% n=50000;
% mu_x=[20;30];
% sigma=[5;6];
% C=[25 -15;
%     -15 36];
% q=500;
% [v,e]=eig(C);
% z=lhscorr(n);
% 
% for i=1:n
%     x(:,i)=mu_x+v*e^0.5*z(:,i);
%     qu(i)=Hansen(x(:,i)');
%     Fs(i,1)=qu(i)/q;
%     if Fs(i,1)<1
%         I(i,1)=1;
%     else
%         I(i,1)=0;
%     end
% end
% pf_m=mean(I)
% pf_std=std(I)/sqrt(n);
% pf_cov=pf_std/pf_m
% toc
% 
% function newz=lhscorr(M)
% for i=1:M
%     z(1,i)=norminv(rand(1,1)/M+(i-1)/M);
%     z(2,i)=norminv(rand(1,1)/M+(i-1)/M);
% end
% [~,o1]=sort(rand(M,1));
% [~,o2]=sort(rand(M,1));
% newz(1,:)=z(1,o1);
% newz(2,:)=z(2,o2);
% end
% 
% function qu=Hansen(x) %汉森公式
% B=2;
% Df=0.5;
% gs=20; %重度
% c=x(1); %粘聚力
% fai=x(2)*pi/180; %标准差6度角度改为弧度
% Nq=(tan(pi/4+fai/2))^2*exp(pi*tan(fai));
% Ng=1.8*(Nq-1)*exp(pi*tan(fai));
% Nc=(Nq-1)*cot(fai);
% qu=0.5*gs*B*Ng+c*Nc+gs*Df*Nq; %承载力
% end

%% Important Sampling

% n=5000;
% mu_x=[20;30];
% C=[25,-15;-15,36];
% q=500;
% cp=[21.54;14.814];
% x=mvnrnd(cp',C,n);
% for i=1:n
%     qu(i,1)=Hansen(x(i,:));
%     Fs(i,1)=qu(i,1)/q;
%     w(i,1)=mvnpdf(x(i,:),mu_x',C)/mvnpdf(x(i,:),cp',C);
%     if Fs(i,1)<1
%         I(i,1)=1;
%     else
%         I(i,1)=0;
%     end
% end
% pf_m=mean(I.*w)
% pf_std=std(I.*w)/sqrt(n);
% pf_cov=pf_std/pf_m
% toc
% 
% function qu=Hansen(x) %汉森公式
% B=2;
% Df=0.5;
% gs=20; %重度
% c=x(1); %粘聚力
% fai=x(2)*pi/180; %标准差6度角度改为弧度
% Nq=(tan(pi/4+fai/2))^2*exp(pi*tan(fai));
% Ng=1.8*(Nq-1)*exp(pi*tan(fai));
% Nc=(Nq-1)*cot(fai);
% qu=0.5*gs*B*Ng+c*Nc+gs*Df*Nq; %承载力
% end

%% Monte Carlo Simulation

% n=5000;
% mu_x=[20;30];
% C=[25,-15;-15,36];
% q=500;
% x=mvnrnd(mu_x,C,n);
% for i=1:n
%     qu(i,1)=Hansen(x(i,:));
%     Fs(i,1)=qu(i,1)/q;
%     if Fs(i,1)<1
%         I(i,1)=1;
%     else
%         I(i,1)=0;
%     end
% end
% pf_m=mean(I)
% pf_cov=sqrt((1-pf_m)/(n*pf_m))
% toc
% 
% function qu=Hansen(x) %汉森公式
% B=2;
% Df=0.5;
% gs=20; %重度
% c=x(1); %粘聚力
% fai=x(2)*pi/180; %标准差6度角度改为弧度
% Nq=(tan(pi/4+fai/2))^2*exp(pi*tan(fai));
% Ng=1.8*(Nq-1)*exp(pi*tan(fai));
% Nc=(Nq-1)*cot(fai);
% qu=0.5*gs*B*Ng+c*Nc+gs*Df*Nq; %承载力
% end

%% generate random vector（多元随机变量）

% mu_x=[20;30];
% sigma_x=[5;6];
% rho=[1 -0.5;-0.5 1]; % 相关情况下
% C=(sigma_x*sigma_x').*rho;
% [a,b]=eig(C);
% mu_y=a'*mu_x;
% sigma_y=sqrt(diag(b));
% 
% for i=1:5000
%     y(1)=normrnd(mu_y(1),sigma_y(1));
%     y(2)=normrnd(mu_y(2),sigma_y(2));
%     x(:,i)=a*y';
% end
% scatter(x(1,:),x(2,:))
% toc

%% generate random vector（多元随机变量）其他函数表达

% mu_x=[20;30];
% sigma_x=[5;6];
% rho=[1 -0.5;-0.5 1]; % 相关情况下
% C=(sigma_x*sigma_x').*rho;
% x=mvnrnd(mu_x',C,5000);
% scatter(x(:,1),x(:,2))
% toc

%% generate random numbers when f(x) is known
% eg.5-2

% a=10;b=20;q=6;r=2;
% f=@(x)(((x-a)^(q-1)*(b-x)^(r-1))/(b-a)^(q+r-1));
% s=@(x)1/(b-a);
% j=0;
% for i=1:5000;
%    x0=unifrnd(a,b);
%    u=rand(1);
%    if u<f(x0)/s(x0)
%        j=j+1;
%        x(j)=x0;
%    end
% end
% bin=[10:0.5:20];
% histogram(x,bin)
% toc

%% generate random numbers when F(x) is known

% for i=1:5000
%     lambda=0.2;
%     u=rand(1,1);
%     x(i)=-log(1-u)/lambda;
% end
% bin=[1:1:50]; 
% histogram(x,bin) % 图表示的是不同的x出现了多少次
% toc

%% rel ana 4-8

% rho=[1 0.6 0 0;0.6 1 0 0;0 0 1 0.4;0 0 0.4 1]; %相关系数矩阵
% 
% mu_x=[100;200;500;10];
% cov_x=[0.3;0.2;0.15;0.1];
% sigma_x=mu_x.*cov_x;
% 
% syms f1 f2 w t
% g=(10.*f1+10.*f2)./(3.*w+5.*t)-1;
% g1=diff(g,f1);
% g2=diff(g,f2);
% g3=diff(g,w);
% g4=diff(g,t);
% 
% x=mu_x; %初始验算点
% y=0;
% yy=1;
% 
% zeta1=sqrt(log(1+power(sigma_x(1)/mu_x(1),2))); % 当量正态化
% zeta2=sqrt(log(1+power(sigma_x(2)/mu_x(2),2)));
% lambda1=log(mu_x(1))-0.5*zeta1^2;
% lambda2=log(mu_x(2))-0.5*zeta2^2;
% beta_T=sqrt(6)/pi*sigma_x(4);
% u=mu_x(4)-0.5772*beta_T;
% 
% i=0;
% while norm(y-yy)>1e-1
%     i=i+1;
%     cdf=[logncdf(x(1),lambda1,zeta1);logncdf(x(2),lambda2,zeta2);normcdf(x(3),mu_x(3),sigma_x(3));gevcdf(x(4),0,beta_T,u)];
% 	pdf=[lognpdf(x(1),lambda1,zeta1);lognpdf(x(2),lambda2,zeta2);normpdf(x(3),mu_x(3),sigma_x(3));gevpdf(x(4),0,beta_T,u)]; 
%     inv_cdf=norminv(cdf);
%     sigma_x1=normpdf(inv_cdf)./pdf;
%     mu_x1=x-inv_cdf.*sigma_x1;
% 
%     f1=x(1); f2=x(2); w=x(3); t=x(4);
%     dgdx=[eval(g1);eval(g2);eval(g3);eval(g4)];
%     
%     C=diag(sigma_x1)*rho*diag(sigma_x1);  %为什么要这么写？？？    
% %     C=rho.*(sigma_x1'*sigma_x1); %协方差矩阵（上个例题这样对，这里不对）
%     [aa,bb]=eig(C); %特征向量矩阵 和 特征值
%     mu_y=aa'*mu_x1;
%     sigma_y=sqrt(diag(bb));
%     
%     y=aa'*x;
%     dgdy=aa'*dgdx;
%     alp=-dgdy.*sigma_y/norm(dgdy.*sigma_y);
% 
%     mu_g=eval(g);
%     beta=(mu_g+dot(dgdy,(mu_y-y)))/norm(dgdy.*sigma_y)
%     yy=y;
%     y=mu_y+beta.*alp.*sigma_y;
%     x=aa*y;
%     
%     z(:,i)=x;
%     betal(i)=beta;
% end
% 
% p=normcdf(-beta,0,1)
% z
% betal

%% rel ana 4-7 （迭代前几步与书上略有差异）

% rho=[1 0.6 0 0;0.6 1 0 0;0 0 1 0.4;0 0 0.4 1]; %相关系数矩阵
% 
% mu_x=[100 200 500 10];
% cov_x=[0.3 0.2 0.15 0.1];
% sigma_x=mu_x.*cov_x;
% % C=rho.*(sigma_x'*sigma_x); %协方差矩阵
% C=diag(sigma_x)*rho*diag(sigma_x);
% [aa,bb]=eig(C); %特征向量矩阵 和 特征值
% mu_y=aa'*mu_x';
% % sigma_y=sqrt(diag(aa'*C*aa));
% sigma_y=sqrt(diag(bb));
% % Nabla_y=aa'*[g1;g2;g3;g4];
% 
% syms f1 f2 w t
% g=(10.*f1+10.*f2)./(3.*w+5.*t)-1;
% g1=diff(g,f1);
% g2=diff(g,f2);
% g3=diff(g,w);
% g4=diff(g,t);
% 
% x0=mu_x; %初始验算点
% x=x0;
% y=aa'*x';
% yy=0;
% i=0;
%     
% while norm(y-yy)>1e-6
%     i=i+1;
%     
%     f1=x(1); f2=x(2); w=x(3); t=x(4);
%     dgdx=[eval(g1) eval(g2) eval(g3) eval(g4)];
%     dgdy=aa'*dgdx';
%     alp=-dgdy.*sigma_y./norm(dgdy.*sigma_y);
% 
%     mu_g=eval(g);
%     beta=(mu_g+dot(dgdy,(mu_y-y)))/norm(dgdy.*sigma_y);
%     yy=y;
%     y=mu_y+beta.*alp.*sigma_y;
%     x=aa*y;
%     
%     ys(:,i)=y;
%     betal(i)=beta;
% end
% 
% p=normcdf(-beta,0,1)
% ys
% betal
% toc

%% rel ana 4-6

% mu_x=[100 200 500 10];
% cov_x=[0.3 0.2 0.15 0.1];
% sigma_x=mu_x.*cov_x;
% 
% syms f1 f2 w t
% g=(10.*f1+10.*f2)./(3.*w+5.*t)-1;
% g1=diff(g,f1);
% g2=diff(g,f2);
% g3=diff(g,w);
% g4=diff(g,t);
% 
% x=mu_x; %初始验算点
% xx=0;
% 
% zeta1=sqrt(log(1+power(sigma_x(1)/mu_x(1),2))); % 当量正态化
% zeta2=sqrt(log(1+power(sigma_x(2)/mu_x(2),2)));
% lambda1=log(mu_x(1))-0.5*zeta1^2;
% lambda2=log(mu_x(2))-0.5*zeta2^2;
% beta_T=sqrt(6)/pi*sigma_x(4);
% u=mu_x(4)-0.5772*beta_T;
% 
% cdf=[logncdf(x(1),lambda1,zeta1) logncdf(x(2),lambda2,zeta2) normcdf(x(3),mu_x(3),sigma_x(3)) gevcdf(x(4),0,beta_T,u)];
% y=norminv(cdf);
%     
% i=0;
% while norm(x-xx)>1e-2
%     i=i+1;
% 	xx=x;
%     pdf=[lognpdf(x(1),lambda1,zeta1) lognpdf(x(2),lambda2,zeta2) normpdf(x(3),mu_x(3),sigma_x(3)) gevpdf(x(4),0,beta_T,u)]; 
%     
%     f1=x(1); f2=x(2); w=x(3); t=x(4);
%     dgdx=[eval(g1) eval(g2) eval(g3) eval(g4)];
%     dgdy=dgdx.*normpdf(y)./pdf;
%     alp=-dgdy./norm(dgdy);
% 
%     beta=(eval(g)-dgdy*y')/norm(dgdy);
%     y=beta.*alp;
%     cdfy=normcdf(y);
%     x=[logninv(cdfy(1),lambda1,zeta1) logninv(cdfy(2),lambda2,zeta2) norminv(cdfy(3),mu_x(3),sigma_x(3)) gevinv(cdfy(4),0,beta_T,u)];
%     
%     z(:,i)=x;
%     betal(i)=beta;
% end
% 
% p=normcdf(-beta,0,1)
% z
% betal

%% rel ana 4-5

% mu_x=[100 200 500 10];
% cov_x=[0.3 0.2 0.15 0.1];
% sigma_x=mu_x.*cov_x;
% 
% syms f1 f2 w t
% g=(10.*f1+10.*f2)./(3.*w+5.*t)-1;
% g1=diff(g,f1);
% g2=diff(g,f2);
% g3=diff(g,w);
% g4=diff(g,t);
% 
% x=mu_x; %初始验算点
% xx=0;
% 
% zeta1=sqrt(log(1+power(sigma_x(1)/mu_x(1),2))); % 当量正态化
% zeta2=sqrt(log(1+power(sigma_x(2)/mu_x(2),2)));
% lambda1=log(mu_x(1))-0.5*zeta1^2;
% lambda2=log(mu_x(2))-0.5*zeta2^2;
% beta_T=sqrt(6)/pi*sigma_x(4);
% u=mu_x(4)-0.5772*beta_T;
% 
% i=0;
% while norm(x-xx)>1e-1
%     i=i+1;
%     
%     pdf=[lognpdf(x(1),lambda1,zeta1) lognpdf(x(2),lambda2,zeta2) normpdf(x(3),mu_x(3),sigma_x(3)) gevpdf(x(4),0,beta_T,u)];
%     cdf=[logncdf(x(1),lambda1,zeta1) logncdf(x(2),lambda2,zeta2) normcdf(x(3),mu_x(3),sigma_x(3)) gevcdf(x(4),0,beta_T,u)];
%     inv_cdf=norminv(cdf);
%     sigma_y=normpdf(inv_cdf)./pdf;
%     mu_y=x-inv_cdf.*sigma_y;
%     
%     f1=x(1); f2=x(2); w=x(3); t=x(4);
%     dgdx=[eval(g1) eval(g2) eval(g3) eval(g4)];
%     alp=-dgdx.*sigma_y./norm(dgdx.*sigma_y);
% 
%     mu_g=eval(g);
%     beta=(mu_g+dot(dgdx,(mu_y-x)))/norm(dgdx.*sigma_y)
%     xx=x;
%     x=mu_y+beta.*alp.*sigma_y;
%     z(:,i)=x;
%     betal(i)=beta;
% end
% 
% p=normcdf(-beta,0,1);
% toc

%% rel ana 4-4 norm dot改进版

% mu_x=[100 200 500 10];
% cov_x=[0.3 0.2 0.15 0.1];
% sigma_x=mu_x.*cov_x;
% 
% syms f1 f2 w t
% g=(10.*f1+10.*f2)./(3.*w+5.*t)-1;
% g1=diff(g,f1);
% g2=diff(g,f2);
% g3=diff(g,w);
% g4=diff(g,t);
% 
% x0=mu_x; %初始验算点
% x=x0;
% xx=0;
%     
% while norm(x-xx)>1e-6
%     f1=x(1); f2=x(2); w=x(3); t=x(4);
%     dgdx=[eval(g1) eval(g2) eval(g3) eval(g4)];
%     alp=-dgdx.*sigma_x./norm(dgdx.*sigma_x);
% 
%     mu_g=eval(g);
%     beta=(mu_g+dot(dgdx,(mu_x-x)))/norm(dgdx.*sigma_x)
%     xx=x;
%     x=mu_x+beta.*alp.*sigma_x;
% end
% 
% p=normcdf(-beta,0,1);
% toc

%% rel ana 4-4

% mu_x=[100 200 500 10];
% cov_x=[0.3 0.2 0.15 0.1];
% sigma_x=mu_x.*cov_x;
% 
% syms f1 f2 w t
% g=(10.*f1+10.*f2)./(3.*w+5.*t)-1;
% g1=diff(g,f1);
% g2=diff(g,f2);
% g3=diff(g,w);
% g4=diff(g,t);
% 
% x0=mu_x; %初始验算点
% x=x0;
% xx=0;
%     
% while norm(x-xx)>1e-3
%     f1=x(1); f2=x(2); w=x(3); t=x(4);
%     sum=power(eval(g1).*sigma_x(1),2)+ ...
%         power(eval(g2).*sigma_x(2),2)+ ...
%         power(eval(g3).*sigma_x(3),2)+ ...
%         power(eval(g4).*sigma_x(4),2);
%     alp1=-eval(g1).*sigma_x(1)./sqrt(sum);
%     alp2=-eval(g2).*sigma_x(2)./sqrt(sum);
%     alp3=-eval(g3).*sigma_x(3)./sqrt(sum);
%     alp4=-eval(g4).*sigma_x(4)./sqrt(sum);
%     alp=[alp1 alp2 alp3 alp4];
% 
%     mu_g=eval(g);
%     beta=(mu_g+eval(g1).*(mu_x(1)-x(1))+ ...
%               eval(g2).*(mu_x(2)-x(2))+ ...
%               eval(g3).*(mu_x(3)-x(3))+ ...
%               eval(g4).*(mu_x(4)-x(4)))./sqrt(sum) 
%     xx=x;
%     x=mu_x+beta.*alp.*sigma_x;
% end
% beta
% p=normcdf(-beta,0,1)
% toc

%% rel ana 4-2

% syms c e h p dp
% 
% g=6.5-c/(1+e).*h.*log10((p+dp)/p);
% 
% g1=diff(g,c);
% g2=diff(g,e);
% g3=diff(g,h);
% g4=diff(g,p);
% g5=diff(g,dp);
% 
% c=0.4; e=1.2; h=500; p=200; dp=30;
% 
% mu_g=eval(g);
% sigma_x=[0.1 0.18 25 10 6];
% sigma_g=sqrt(power(eval(g1).*sigma_x(1),2)+power(eval(g2).*sigma_x(2),2)+power(eval(g3).*sigma_x(3),2)+power(eval(g4).*sigma_x(4),2)+power(eval(g5).*sigma_x(5),2));
% 
% beta=mu_g/sigma_g;
% p=normcdf(-beta,0,1)
% toc

%% 可靠度 例4-1

% mu_x=[100 200 500 10];
% delta_x=[0.3 0.2 0.15 0.1];
% 
% sigma_x=mu_x.*delta_x;
% mu_g=10*mu_x(1)+10*mu_x(2)-3*mu_x(3)-5*mu_x(4);
% sigma_g=sqrt(power(10*sigma_x(1),2)+power(10*sigma_x(2),2)+power((-3)*sigma_x(3),2)+power((-5)*sigma_x(4),2));
% 
% beta=mu_g/sigma_g;
% p=normcdf(-beta,0,1)
% toc


%% 统计特性函数

%Example 2.11
% weight=[16.80 19.04 18.72 15.84 16.16 15.36 18.24 16.00 ...
%     15.84 16.32 16.00 16.16 16.16 16.00 16.16 15.84 ...
%     16.00 16.48 16.16 16.96 17.44 16.00 16.64 16.32 ...
%     16.96 15.84 16.32 16.00 16.16 16.16 16.64 16.32 ...
%     16.80 15.20 18.56 17.12 17.92 18.24 17.44 17.60 ...
%     17.44 16.96 17.28 17.76 20.00 17.92 16.64 18.08 ...
%     17.92 18.08 18.56 19.84 18.72 18.24 18.40 18.24 ...
%     18.40 18.24 17.92 18.40 18.40 17.92 18.40 19.04 ...
%     ];
% w_mean = mean(weight)
% w_mode = mode(weight)
% w_median = median(weight)
% w_std = std(weight)
% w_var = var(weight)
% w_skew = skewness(weight)
% w_kurt = kurtosis(weight)


% %Example 2.12
% A=[2.82 1.32;2.97 1.02;4.55 2.46;14.27 7.42;2.87 0.43
%     3.91 0.48;8.10 1.93;4.39 1.68;5.31 1.98;6.99 3.15
%     3.05 0.99;2.57 0.76;4.17 1.78;3.99 1.96;3.91 1.50
%     5.31 2.41;8.99 2.59;2.97 0.99;2.92 0.58;6.53 1.14
%     9.07 4.04;12.98 4.42;3.86 1.42;7.44 2.84;2.95 1.63];
% A_cov=cov(A)
% A_cov1=cov(A,1)
% A_cor=corrcoef(A)

%% 新建多个文件夹并移动文件进去 的 小应用

% content_struct=dir('C:\Users\Dell\Downloads\新建文件夹\《MATLAB 神经网络43个案例分析》源代码&数据\'); %以struct形式列出文件夹内容
% content_cell=struct2cell(content_struct);
% content=content_cell(1,3:45); % 这里存在自然排序的问题
% 
% for i=1:43
%     content1=content{1,i}(1:end-4);
%     
%     folder = 'C:\Users\Dell\Downloads\新建文件夹\《MATLAB 神经网络43个案例分析》源代码&数据\'; % new_folder 保存要创建的文件夹，是绝对路径+文件夹名称
%     new_folder=[folder,content1(1,8:end)];
%     mkdir(new_folder);  % mkdir()函数创建文件夹
%     path1=[folder,content{i}];
%     path2=new_folder;
%     movefile(path1,path2); % 移动文件
% end

%% 雷克子波

% t=-2:0.01:2;
% fm=1;
% a=1;
% a=(1-2*(pi*fm*t).^2).*exp(-(pi*fm*t).^2);
% 
% plot(a)

%% logarithmic spiral

% fai=0.01:0.1:200; %Polar coordinates
% r=exp(fai);
% polar(fai,r)
% 
% fai=0.01:0.1:200; %Cartesian coordinates
% for i=1:length(fai)
%     x(i)=exp(fai(i))*cos(fai(i));
%     y(i)=exp(fai(i))*sin(fai(i));
% end
% plot(y,x);
% 
% % %用复数
% % fai=0.01:0.1:200;
% % z=exp((1+i)*fai); %complex number
% % plot(z)

