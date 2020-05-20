######################################################################
#Plotting data and results of classification
######################################################################

import plotly.graph_objects as go


def plot_features(df1,df2,ptitle,xtitle,ytitle):
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df1["C1"], y=df2["C1"],
                    mode='markers',
                    name='Class - C1'))
    fig.add_trace(go.Scatter(x=df1["C2"], y=df2["C2"],
                    mode='markers',
                    name='Class - C2'))
    fig.add_trace(go.Scatter(x=df1["C3"], y=df2["C3"],
                    mode='markers',
                    name='Class - C3'))
    fig.add_trace(go.Scatter(x=df1["C4"], y=df2["C4"],
                    mode='markers',
                    name='Class - C4'))
    fig.add_trace(go.Scatter(x=df1["C5"], y=df2["C5"],
                    mode='markers',
                    name='Class - C5'))
    
    fig.update_layout(title=ptitle,xaxis_title=xtitle,yaxis_title=ytitle)
    fig.show()
    
def plot_corwrgpreds(cor_preds,wrong_preds,adtitle):
    
    classes = ['C1','C2','C3','C4','C5']

    fig = go.Figure()
    fig.add_trace(go.Bar(x=classes,
                    y=cor_preds,
                    name='Correctly Predicted',
                    marker_color='#00CC96'
                    ))
    fig.add_trace(go.Bar(x=classes,
                    y=wrong_preds,
                    name='Incorrectly Predicted',
                    marker_color='#EF553B'
                    ))
    
    fig.update_layout(
        title='Correct and Wrong Predictions per class'+adtitle,
        xaxis_tickfont_size=14,
        yaxis=dict(
            title='Number of predictions',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap=0.15, 
        bargroupgap=0.1 
    )
    
    fig.show()
    
    
def plot_accerror(acc,err,adtitle=""):
    
    cases = ['F1','Z1','F2','[Z1:F2]']

    fig = go.Figure()
    fig.add_trace(go.Bar(x=cases,
                    y=acc,
                    name='Accuracy',
                    marker_color='#00CC96'
                    ))
    fig.add_trace(go.Bar(x=cases,
                    y=err,
                    name='Error',
                    marker_color='#EF553B'
                    ))
    
    fig.update_layout(
        title='Accuracy v/s Error for all cases'+adtitle,
        xaxis_tickfont_size=14,
        yaxis=dict(
            title='Accuracy or Error',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap=0.15, 
        bargroupgap=0.1 
    )
    
    fig.show()
    
    