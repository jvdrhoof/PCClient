import math

import help_functions as hf


class RateAdaptation:
    def __init__(self, n_qualities, rah, predict):
        self.n_qualities = n_qualities
        self.rah = rah
        self.predict = predict

    def __repr__(self):
        return f'PC rate adaptation {self.rah}'

    def __str__(self):
        return f'PC rate adaptation {self.rah}'

    def adapt(self, frame_start, gop, frame_played, budget, data):
        print(frame_start, gop, frame_played, budget)

        if budget == 0:
            return [1] * len(set(data['pc'].tolist()))

        if self.predict == 0:
            e = data.loc[data['frame'] == frame_played]
        else:
            e = data.loc[data['frame'] == frame_start]
    
        # Assign bandwidth according to distance
        if self.rah == 1:
            i = e.sort_values(by=['dist'])
    
        # Assign bandwidth according to visible area, then distance
        elif self.rah == 2:
            i = e.sort_values(by=['A_vis', 'dist'], ascending=[0, 1])
        
        # Assign bandwidth according to visible area, then potential area, then distance
        elif self.rah == 3:
            i = e.sort_values(by=['A_vis', 'A_pot', 'dist'], ascending=[0, 0, 1])
        
        # Assign bandwidth according to potential area, then distance
        elif self.rah == 4:
            i = e.sort_values(by=['A_pot', 'dist'], ascending=[0, 1])
        
        # Assign bandwidth according to potential area, then visible area, then distance
        elif self.rah == 5:
            i = e.sort_values(by=['A_pot', 'A_vis', 'dist'], ascending=[0, 0, 1])
        
        # Assign bandwidth according to visible area / points, then distance
        elif self.rah == 6:
            e['A_vis_points'] = e['A_vis'] / e['points']
            i = e.sort_values(by=['A_vis_points', 'dist'], ascending=[0, 1])
        
        # Assign bandwidth according to potential area / points, then distance
        else:
            e['A_pot_points'] = e['A_pot'] / e['points']
            i = e.sort_values(by=['A_pot_points', 'dist'], ascending=[0, 1])
    
        #print(i)
        j = i['pc'].tolist()
        #print(j)
        
        f = data.loc[(data['frame'] >= frame_start) & (data['frame'] < frame_start + gop)]
        #print(f)
        g = f.groupby(by=['pc'], as_index=False)
        #print(g)
        h = g.mean()
        #print(h)
        
        bits_min = sum(h['r1'].tolist()) * gop
        print(bits_min)
        
        reps = [1] * len(j)
        if bits_min > budget:
            return reps
        
        bits = bits_min
        for pc in j:
            for r in range(2, self.n_qualities + 1):
                e = h.loc[h['pc'] == pc]
                cost = (e.iloc[0]['r%i' % r] - e.iloc[0]['r%i' % (r - 1)]) * gop
                if bits + cost <= budget:
                    reps[j.index(pc)] += 1
                    bits += cost
                else:
                    break
        print(bits)
    
        return reps
