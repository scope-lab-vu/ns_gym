# AAMAS 2026 Competition Leaderboard

This leaderboard mirrors the four competition criteria described in [Evaluating Your Agent](aamas2026_competition/evaluation.md).

```{eval-rst}
.. note::

   Public rankings are populated from organizer-run evaluations. The tables below will be filled as scored submissions are verified and posted.
```
## Adaptability

<div class="leaderboard-card">
  <div class="leaderboard-badges">
    <span class="leaderboard-badge">un-notify</span>
  </div>
  <div class="leaderboard-table-wrap">
    <table class="leaderboard-table">
      <thead>
        <tr>
          <th scope="col">Rank</th>
          <th scope="col">GitHub ID</th>
          <th scope="col">Approach (3 experiments)</th>
          <th scope="col">Threshold Success</th>
          <th scope="col">Convergence Time</th>
          <th scope="col">Local Stability</th>
        </tr>
      </thead>
      <tbody>
          <tr>
          <td>1</td>
          <td>Jazhyc</td>
          <td>Context-Aware PPO/PPO/PPO</td>
          <td>0.711111</td>
          <td>0.332233</td>
          <td>0.000066</td>
        </tr>
          <tr>
          <td>2</td>
          <td>pudovikoff</td>
          <td>MLE-Replan/DQN/SAC</td>
          <td>0.633333</td>
          <td>0.366667</td>
          <td>0.366732</td>
        </tr>
        <tr>
          <td>3</td>
          <td>Kumar20-21</td>
          <td>Dynamic VI/LQR/QR-SAC</td>
          <td>0.666667</td>
          <td>0.333333</td>
          <td>0.111177</td>
        </tr>
        <tr>
          <td>4</td>
          <td>micahr234</td>
          <td>MOUSE+sequence model+history/MOUSE/-</td>
          <td>0.666667</td>
          <td>0.333333</td>
          <td>0.333400</td>
        </tr>
        <tr>
          <td>5</td>
          <td>micahr234</td>
          <td>MOUSE+MLP+no history/MOUSE/-</td>
          <td>0.666667</td>
          <td>0.333333</td>
          <td>0.333400</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

## Performance


```{raw} html
<div class="leaderboard-card">
  <div class="leaderboard-tabs">
    <input type="radio" id="performance-unnotify" name="performance-mode" checked>
    <label for="performance-unnotify">Un-notify</label>

    <input type="radio" id="performance-partial" name="performance-mode">
    <label for="performance-partial">Partial-notify</label>

    <div class="leaderboard-tab-panels">
      <section class="leaderboard-panel">
        <div class="leaderboard-table-wrap">
          <table class="leaderboard-table">
            <thead>
              <tr>
                <th scope="col">Rank</th>
                <th scope="col">GitHub ID</th>
                <th scope="col">Approach (3 experiments)</th>
                <th scope="col">Average return</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>1</td>
                <td>Jazhyc</td>
                <td>Context-Aware PPO/PPO/PPO</td>
                <td>2257.240</td>
              </tr>
              <tr>
                <td>2</td>
                <td>pudovikoff</td>
                <td>MLE-Replan/DQN/SAC</td>
                <td>1800.111</td>
              </tr>
              <tr>
                <td>3</td>
                <td>Kumar20-21</td>
                <td>Dynamic VI/LQR/QR-SAC</td>
                <td>1734.251</td>
              </tr>
              <tr>
                <td>4</td>
                <td>micahr234</td>
                <td>MOUSE+sequence model+history/MOUSE/-</td>
                <td>568.996</td>
              </tr>
              <tr>
                <td>5</td>
                <td>micahr234</td>
                <td>MOUSE+MLP+no history/MOUSE/-</td>
                <td>567.229</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section class="leaderboard-panel">
        <div class="leaderboard-table-wrap">
          <table class="leaderboard-table">
            <thead>
              <tr>
                <th scope="col">Rank</th>
                <th scope="col">GitHub ID</th>
                <th scope="col">Approach (3 experiments)</th>
                <th scope="col">Average return</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>1</td>
                <td>Jazhyc</td>
                <td>Context-Aware PPO/PPO/PPO</td>
                <td>2197.415</td>
              </tr>
              <tr>
                <td>2</td>
                <td>pudovikoff</td>
                <td>MLE-Replan/DQN/SAC</td>
                <td>1800.111</td>
              </tr>
              <tr>
                <td>3</td>
                <td>Kumar20-21</td>
                <td>Dynamic VI/LQR/QR-SAC</td>
                <td>1734.251</td>
              </tr>
              <tr>
                <td>4</td>
                <td>micahr234</td>
                <td>MOUSE+sequence model+history/MOUSE/-</td>
                <td>568.996</td>
              </tr>
              <tr>
                <td>5</td>
                <td>micahr234</td>
                <td>MOUSE+MLP+no history/MOUSE/-</td>
                <td>567.229</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  </div>
</div>
```

## Resilience

<div class="leaderboard-card">
  <div class="leaderboard-badges">
    <span class="leaderboard-badge">fully-notify</span>
  </div>
  <div class="leaderboard-table-wrap">
    <table class="leaderboard-table">
      <thead>
        <tr>
          <th scope="col">Rank</th>
          <th scope="col">GitHub ID</th>
          <th scope="col">Approach (3 experiments)</th>
          <th scope="col">Immediate Drop</th>
          <th scope="col">Recovery Time</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>1</td>
          <td>Jazhyc</td>
          <td>Context-Aware PPO/PPO/PPO</td>
          <td>0.333333</td>
          <td>0.332727</td>
        </tr>
        <tr>
          <td>2</td>
          <td>Kumar20-21</td>
          <td>Dynamic VI/LQR/QR-SAC</td>
          <td>0.333333</td>
          <td>0.333333</td>
        </tr>
        <tr>
          <td>3</td>
          <td>micahr234</td>
          <td>MOUSE+sequence model+history/MOUSE/-</td>
          <td>0.333333</td>
          <td>0.333333</td>
        </tr>
        <tr>
          <td>4</td>
          <td>micahr234</td>
          <td>MOUSE+MLP+no history/MOUSE/-</td>
          <td>0.333333</td>
          <td>0.333333</td>
        </tr>
        <tr>
          <td>5</td>
          <td>pudovikoff</td>
          <td>MLE-Replan/DQN/SAC</td>
          <td>0.366667</td>
          <td>0.366667</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

## Efficiency

```{raw} html
<div class="leaderboard-card">
  <div class="leaderboard-tabs">
    <input type="radio" id="efficiency-unnotify" name="efficiency-mode" checked>
    <label for="efficiency-unnotify">Un-notify</label>

    <input type="radio" id="efficiency-partial" name="efficiency-mode">
    <label for="efficiency-partial">Partial-notify</label>

    <div class="leaderboard-tab-panels">
      <section class="leaderboard-panel">
        <div class="leaderboard-table-wrap">
          <table class="leaderboard-table">
            <thead>
              <tr>
                <th scope="col">Rank</th>
                <th scope="col">GitHub ID</th>
                <th scope="col">Approach (3 experiments)</th>
                <th scope="col">Efficiency</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>1</td>
                <td>Jazhyc</td>
                <td>Context-Aware PPO/PPO/PPO</td>
                <td>0.00001430</td>
              </tr>
              <tr>
                <td>2</td>
                <td>Kumar20-21</td>
                <td>Dynamic VI/LQR/QR-SAC</td>
                <td>0.00001862</td>
              </tr>
              <tr>
                <td>3</td>
                <td>pudovikoff</td>
                <td>MLE-Replan/DQN/SAC</td>
                <td>0.00005887</td>
              </tr>
              <tr>
                <td>4</td>
                <td>micahr234</td>
                <td>MOUSE+MLP+no history/MOUSE/-</td>
                <td>0.00253916</td>
              </tr>
              <tr>
                <td>5</td>
                <td>micahr234</td>
                <td>MOUSE+sequence model+history/MOUSE/-</td>
                <td>0.08414263</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section class="leaderboard-panel">
        <div class="leaderboard-table-wrap">
          <table class="leaderboard-table">
            <thead>
              <tr>
                <th scope="col">Rank</th>
                <th scope="col">GitHub ID</th>
                <th scope="col">Approach (3 experiments)</th>
                <th scope="col">Efficiency</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>1</td>
                <td>Jazhyc</td>
                <td>Context-Aware PPO/PPO/PPO</td>
                <td>0.00000918</td>
              </tr>
              <tr>
                <td>2</td>
                <td>Kumar20-21</td>
                <td>Dynamic VI/LQR/QR-SAC</td>
                <td>0.00001913</td>
              </tr>
              <tr>
                <td>3</td>
                <td>pudovikoff</td>
                <td>MLE-Replan/DQN/SAC</td>
                <td>0.00005943</td>
              </tr>
              <tr>
                <td>4</td>
                <td>micahr234</td>
                <td>MOUSE+MLP+no history/MOUSE/-</td>
                <td>0.00323941</td>
              </tr>
              <tr>
                <td>5</td>
                <td>micahr234</td>
                <td>MOUSE+sequence model+history/MOUSE/-</td>
                <td>0.08204777</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  </div>
</div>
```