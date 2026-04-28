classdef Policy < handle
    % Policy  使用A*算法的简单全局路径规划策略
    %
    % 输出动作 action = [u, w]
    %   u: 线速度(正向前进)
    %   w: 角速度(朝向下一路径点转向)
    %
    % 依赖：observation.map (全局占据栅格), observation.agent (含 x,y,h),
    %      observation.endPos (目标点)
    %
    % 说明：仓库里的 AStar.m 会在构造函数里画图且没有返回路径接口，
    %      因此这里在 Policy.m 内部实现一个“无绘图”的A*，仅用于生成路径。

    properties
        path            % Nx2 路径点 [x y]
        pathIdx (1,1) double = 1

        lastGoal        % [x y]
        lastStart       % [x y]
        lastMapHash

        % 控制参数
        kp (1,1) double = 2.0        % 角速度比例
        goalTol (1,1) double = 0.8   % 到达路径点的距离阈值(格)
        lookAhead (1,1) double = 1.5 % 前视距离(格)

        % 速度限制（应与 Agent.setSatLevel 一致或更小）
        uMax (1,1) double = 10
        wMax (1,1) double = 1

        % A*参数
        allowDiagonal (1,1) logical = true
    end

    methods
        function self = Policy()
        end

        function action = action(self, observation)
            % 发生碰撞时，先做轻微脱困（保持与你原始Policy一致的行为风格）
            if isfield(observation, 'collide') && observation.collide
                action = [-10, rand(1) - 0.5];
                return;
            end

            % 若没有全局地图（globalview=false），退化为原始随机转向的直行策略
            if ~isfield(observation, 'map') || isempty(observation.map)
                action = [10, rand(1) - 0.5];
                return;
            end

            % 当前状态
            agent = observation.agent;
            sx = round(agent.x);
            sy = round(agent.y);
            sh = agent.h;

            % 目标
            gx = round(observation.endPos.x);
            gy = round(observation.endPos.y);

            occ = observation.map; % 注意：Observation里已做转置，map(y,x)

            % A*重规划判定
            needReplan = self.needReplan([sx sy], [gx gy], occ);
            if needReplan
                self.path = self.planAStar([sx sy], [gx gy], occ);
                self.pathIdx = 1;

                self.lastStart = [sx sy];
                self.lastGoal  = [gx gy];
                self.lastMapHash = self.hashMap(occ);
            end

            % 无路可走：原地小幅随机转向 + 慢速前进
            if isempty(self.path)
                action = [3, rand(1) - 0.5];
                return;
            end

            % 推进路径索引：如果已经接近当前路径点，跳到下一个
            while self.pathIdx < size(self.path, 1) && ...
                    norm([agent.x agent.y] - self.path(self.pathIdx, :)) < self.goalTol
                self.pathIdx = self.pathIdx + 1;
            end

            target = self.selectLookAheadTarget([agent.x agent.y], self.path, self.pathIdx, self.lookAhead);

            % 追踪控制（朝向target）
            dx = target(1) - agent.x;
            dy = target(2) - agent.y;
            desired = atan2(dy, dx);
            err = self.wrapToPi(desired - sh);

            % 角速度
            w = self.kp * err;
            w = max(min(w, self.wMax), -self.wMax);

            % 线速度：当偏航大时降速
            u = self.uMax * max(0.2, cos(err));
            u = max(min(u, self.uMax), -self.uMax);

            action = [u, w];
        end
    end

    methods (Access = private)
        function tf = needReplan(self, start, goal, occ)
            tf = false;
            if isempty(self.path) || self.pathIdx > size(self.path, 1)
                tf = true; return;
            end
            if isempty(self.lastGoal) || any(self.lastGoal ~= goal)
                tf = true; return;
            end

            % 起点变化太大也触发
            if isempty(self.lastStart) || norm(self.lastStart - start) > 2
                tf = true; return;
            end

            % 地图变化触发（简单hash）
            h = self.hashMap(occ);
            if isempty(self.lastMapHash) || self.lastMapHash ~= h
                tf = true; return;
            end

            % 当前路径点若变成障碍也触发
            idx = min(self.pathIdx, size(self.path,1));
            p = self.path(idx, :);
            if self.isBlocked(round(p(1)), round(p(2)), occ)
                tf = true; return;
            end
        end

        function h = hashMap(~, occ)
            % 低成本hash：对障碍位置求和（足够用于是否变化的启发）
            occ = occ ~= 0;
            [yy, xx] = find(occ);
            h = uint64(0);
            if ~isempty(xx)
                h = uint64(sum(uint64(xx) .* 1315423911 + uint64(yy) .* 2654435761));
            end
        end

        function blocked = isBlocked(~, x, y, occ)
            if x < 1 || y < 1 || y > size(occ,1) || x > size(occ,2)
                blocked = true; return;
            end
            blocked = occ(y, x) ~= 0;
        end

        function path = planAStar(self, start, goal, occ)
            % start/goal: [x y]，occ: map(y,x) 非0为障碍
            % 返回：Nx2 [x y]，包含起点和终点；若失败返回[]

            % 若起点/终点不可达，直接失败
            if self.isBlocked(start(1), start(2), occ) || self.isBlocked(goal(1), goal(2), occ)
                path = [];
                return;
            end

            [H, W] = size(occ);
            start = double(start);
            goal  = double(goal);

            % 8邻域 or 4邻域
            if self.allowDiagonal
                neigh = [ -1  0; 1  0; 0 -1; 0  1; -1 -1; -1 1; 1 -1; 1 1];
                stepCost = [1;1;1;1;sqrt(2);sqrt(2);sqrt(2);sqrt(2)];
            else
                neigh = [ -1  0; 1  0; 0 -1; 0  1];
                stepCost = [1;1;1;1];
            end

            % gScore & fScore
            gScore = inf(H, W);
            fScore = inf(H, W);

            % parent: 用线性索引存父节点线性索引（0表示无）
            parent = zeros(H, W, 'uint32');

            sx = start(1); sy = start(2);
            gx = goal(1);  gy = goal(2);

            gScore(sy, sx) = 0;
            fScore(sy, sx) = self.heuristic([sx sy], [gx gy]);

            % open set: 用简单数组实现优先队列 (f最小)
            open = zeros(0, 2);
            open = [open; sx sy];
            inOpen = false(H, W);
            inOpen(sy, sx) = true;
            closed = false(H, W);

            found = false;
            while ~isempty(open)
                % 取f最小
                fs = arrayfun(@(i) fScore(open(i,2), open(i,1)), 1:size(open,1));
                [~, bestIdx] = min(fs);
                current = open(bestIdx, :);
                open(bestIdx, :) = [];
                cx = current(1); cy = current(2);
                inOpen(cy, cx) = false;

                if cx == gx && cy == gy
                    found = true;
                    break;
                end

                closed(cy, cx) = true;

                for k = 1:size(neigh,1)
                    nx = cx + neigh(k,1);
                    ny = cy + neigh(k,2);

                    if nx < 1 || ny < 1 || nx > W || ny > H
                        continue;
                    end
                    if closed(ny, nx)
                        continue;
                    end
                    if occ(ny, nx) ~= 0
                        continue;
                    end

                    % 若斜走，避免“穿角”
                    if self.allowDiagonal && abs(neigh(k,1))==1 && abs(neigh(k,2))==1
                        if occ(cy, nx) ~= 0 || occ(ny, cx) ~= 0
                            continue;
                        end
                    end

                    tentativeG = gScore(cy, cx) + stepCost(k);
                    if tentativeG < gScore(ny, nx)
                        gScore(ny, nx) = tentativeG;
                        fScore(ny, nx) = tentativeG + self.heuristic([nx ny], [gx gy]);
                        parent(ny, nx) = uint32(sub2ind([H W], cy, cx));
                        if ~inOpen(ny, nx)
                            open = [open; nx ny];
                            inOpen(ny, nx) = true;
                        end
                    end
                end
            end

            if ~found
                path = [];
                return;
            end

            % 回溯路径
            path = zeros(0,2);
            cx = gx; cy = gy;
            path = [cx cy; path];
            while ~(cx == sx && cy == sy)
                p = parent(cy, cx);
                if p == 0
                    path = []; return; % 理论上不应发生
                end
                [py, px] = ind2sub([H W], double(p));
                cx = px; cy = py;
                path = [cx cy; path];
            end
        end

        function h = heuristic(~, a, b)
            % 欧式距离作为启发
            dx = a(1) - b(1);
            dy = a(2) - b(2);
            h = sqrt(dx*dx + dy*dy);
        end

        function target = selectLookAheadTarget(~, pos, path, idx, lookAhead)
            % 从idx开始，在路径上找一个距离pos至少lookAhead的点
            target = path(min(idx, size(path,1)), :);
            for i = idx:size(path,1)
                if norm(path(i,:) - pos) >= lookAhead
                    target = path(i,:);
                    return;
                end
            end
            % 否则取终点
            target = path(end,:);
        end

        function a = wrapToPi(~, a)
            a = mod(a + pi, 2*pi) - pi;
        end
    end
end
