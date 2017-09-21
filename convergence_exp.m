timestr = datestr(now, 'yyyymmdd_HHMM');
system(['mkdir ' timestr]);
system(['mkdir ' timestr '\qqplot']);
system(['mkdir ' timestr '\nmseplot']);
kappa_start = 2.2;
kappa_end   = 15;
kappa_step  = 0.1;
qq_plot   = figure('visible', 'off');
nmse_plot = figure('visible', 'off');
step = 0;
nsteps = (kappa_end - kappa_start) / kappa_step;
amp_conv_list   = zeros(1, nsteps);
fista_conv_list = zeros(1, nsteps);
ista_conv_list  = zeros(1, nsteps);

for kappa = kappa_start:kappa_step:kappa_end
    command = sprintf('python gen_kappa.py %.1f', kappa);
    file_name = sprintf('problem_k%.1f.mat', kappa);
    % count the numbers of convergence of amp, fista and ista
    amp_conv   = 0;
    fista_conv = 0;
    ista_conv  = 0;
    for i = 1:100
        fprintf('kappa = %.1f  iteration %03d\n', kappa, i);
        system(command);

        load(file_name);
        
        clf(qq_plot);
        clf(nmse_plot);

        amp_conv_flag   = 0;
        fista_conv_flag = 0;
        ista_conv_flag  = 0;

        L = size(x,2);
        [M,N] = size(A);
        Afro2 = norm(A,'fro')^2;

        assert(abs(Afro2-N)/N < 0.01,'I was assuming this would have unit norm columns')
        z = A*x;
        wvar =  mean((y(:)-z(:)).^2);
        SNRdB_test = 10*log10(mean(abs(z(:)).^2)/wvar);
        supp = x(:)~=0;
        K = mean(supp)*N;

        % algorithm parameters
        T = 1e2; % AMP iterations
        Ti = 1e3; % FISTA iterations
        Tii = 1e4; % ISTA iterations
        alf = 1.1402; % amp tuning parameter [1.1402]
        nmse_dB_report = -35;
        eta = @(r,lam) sign(r).*max(bsxfun(@minus,abs(r),lam),0);
        tqq = 10; tqqi = 100; tqqii = 1000; % iteration for qqplot

        % run AMP
        Bmf = A'; % matched filter
        xhat = zeros(N,L); % initialization of signal estimate
        v = zeros(M,L); % initialization of residual
        nmse_amp = [ones(1,L);zeros(T,L)];
        qq = true;
        report = true;
        for t=1:T
          g = (N/M)*mean(xhat~=0,1); % onsager gain
          v = y - A*xhat + bsxfun(@times,v,g); % residual
          rhat = xhat + Bmf*v; % denoiser input
          rvar = sum(abs(v).^2,1)/M; % denoiser input err var
          xhat = eta(rhat, alf*sqrt(rvar)); % estimate
          nmse_amp(t+1,:) = sum(abs(xhat-x).^2,1)./sum(abs(x).^2,1);
          if qq&(mean(nmse_amp(t+1,:))<0.1),
            set(0, 'CurrentFigure', qq_plot)
            subplot(131);
            qqplot(rhat(:,1)-x(:,1));
            axis('square')
            title(['AMP at iteration ',num2str(t)]);
            drawnow;
            qq = false;
          end
          if report&&(mean(nmse_amp(t+1,:))<10^(nmse_dB_report/10)),
            fprintf('AMP reached NMSE=%ddB at iteration %i\n',nmse_dB_report,t);
            report = false;
          end
        end
        if mean(nmse_amp(end,:)) > .1
            lam_mf = .01;
            fprintf('AMP did not converge! ... using a wild guess lam_mf=%f for ISTA,FISTA\n',lam_mf);
        else
            xhat_mf = xhat;
            fprintf('AMP terminal NMSE=%.4f dB\n', 10*log10(mean(nmse_amp(end,:))) );
            lam_mf = alf*sqrt(sum(abs(v).^2,1)/M).*(1-sum(xhat~=0,1)/M); % lam for lasso
            amp_conv_flag = 1;
            amp_conv = amp_conv + 1;
            %lam_mf = max(abs(Bmf*(y-A*xhat))); % another way to compute lam for lasso
        end

        % run FISTA
        scale = .999/norm(Bmf*A);
        B = scale*Bmf;
        xhat = zeros(N,L); % initialization of signal estimate
        xhat_old = zeros(N,L);
        nmse_fista = [ones(1,L);zeros(Ti,L)];
        qq = true;
        report = true;
        for t=1:Ti
          v = y - A*xhat; % residual
          rhat = xhat + B*v + ((t-2)/(t+1))*(xhat-xhat_old); % denoiser input
          xhat_old = xhat;
          xhat = eta(rhat, lam_mf*scale); % estimate
          nmse_fista(t+1,:) = sum(abs(xhat-x).^2,1)./sum(abs(x).^2,1);
          if qq&&(mean(nmse_fista(t+1,:))<0.1),
            %figure(2)
            subplot(132);
            qqplot(rhat(:,1)-x(:,1));
            axis('square')
            title(['FISTA at iteration ',num2str(t)]);
            drawnow;
            qq = false;
          end
          if report&&(mean(nmse_fista(t+1,:))<10^(nmse_dB_report/10)),
            fprintf('FISTA reached NMSE=%ddB at iteration %i\n',nmse_dB_report,t);
            report = false;
          end
        end
        if mean(nmse_fista(end, :)) > .1
            fprintf('FISTA did not converge! ...\n')
        else
            xhat_fista_mf = xhat;
            fprintf('FISTA terminal NMSE=%.4f dB\n', 10*log10(mean(nmse_fista(end,:))) );
            lam_mf_test = max(abs(Bmf*(y-A*xhat))); % another way to compute lam for lasso
            fista_conv_flag = 1;
            fista_conv = fista_conv + 1;
        end

        % run ISTA
        xhat = zeros(N,L); % initialization of signal estimate
        nmse_ista = [ones(1,L);zeros(Tii,L)];
        qq = true;
        report = true;
        for t=1:Tii
          v = y - A*xhat; % residual
          rhat = xhat + B*v; % denoiser input
          xhat = eta(rhat, lam_mf*scale); % estimate
          nmse_ista(t+1,:) = sum(abs(xhat-x).^2,1)./sum(abs(x).^2,1);
          if qq&&(mean(nmse_ista(t+1,:))<0.1),
            %figure(2)
            subplot(133);
            qqplot(rhat(:,1)-x(:,1));
            axis('square')
            title(['ISTA at iteration ',num2str(t)]);
            drawnow;
            qq = false;
          end
          if report&&(mean(nmse_ista(t+1,:))<10^(nmse_dB_report/10)),
            fprintf('ISTA reached NMSE=%ddB at iteration %i\n',nmse_dB_report,t);
            report = false;
          end
        end
        if mean(nmse_ista(end, :)) > .1
            fprintf('ISTA did not converge! ...\n');
        else
            xhat_ista_mf = xhat;
            fprintf('ISTA terminal NMSE=%.4f dB\n', 10*log10(mean(nmse_ista(end,:))) );
            lam_mf_test = max(abs(Bmf*(y-A*xhat))); % another way to compute lam for lasso
            ista_conv_flag = 1;
            ista_conv = ista_conv + 1;
        end

        % plot results
        if amp_conv_flag && fista_conv_flag && ista_conv_flag
            % save qq plot to "$time/qqplot/qq_k$kappa_i.png"
            qqplot_file = sprintf('%s/qqplot/qq_k%2.1f_%03d.png', timestr, kappa, i);
            saveas(qq_plot, qqplot_file)

            % draw and save nmse plot
            nmseplot_name = sprintf('%s/nmseplot/nmse_k%2.1f_%03d.png', timestr, kappa, i);
            set(0, 'CurrentFigure', nmse_plot);
            handy = semilogx([0:Tii],10*log10(mean(nmse_ista,2)),'b.-',...
                     [0:Ti],10*log10(mean(nmse_fista,2)),'g.-',...
                     [0:T],10*log10(mean(nmse_amp,2)),'k.-');
            set(handy(2),'Color',[0 0.5 0])
            legend('ISTA','FISTA','AMP')
            ylabel('NMSE [dB]')
            xlabel('iterations')
            grid on
            title(['N=',num2str(N),', M=',num2str(M),', E[K]=',num2str(K),', SNRdB=',num2str(SNRdB_test),', MMV=',num2str(L)])
            saveas(nmse_plot, nmseplot_name);
        end
    end
    step = step + 1;
    amp_conv_list(step) = amp_conv;
    fista_conv_list(step) = fista_conv;
    ista_conv_list(step) = ista_conv;
end

conv_prob_plot = figure('visible', 'off');
set(0, 'CurrentFigure', conv_prob_plot);
x = kappa_start:kappa_step:kappa_end;
plot(x, amp_conv_list, x, fista_conv_list, x, ista_conv_list);
legend('AMP', 'FISTA', 'AMP')
ylabel('Convergence Probability')
xlabel('kappa')
grid on
conv_prob_plot_name = sprintf('%s/conv_prob.png', timestr);
saveas(conv_prob_plot, conv_prob_plot_name)
