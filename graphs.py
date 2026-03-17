import matplotlib as plt

class Graphs:
    
    BARH = 1
    BAR = 2
    PIE = 3
    
    def __init__(self, px_data = None,py_data=None,
                 pwidth= 10,pheight= 6,pcolor = 'coral',
                 pylabel='',pxlabel='',ptitle = ''):
        self.title  = ptitle
        self.xlabel = pxlabel
        self.x_data = px_data
        self.y_data = py_data
        self.ylabel = pylabel
        self.width  = pwidth
        self.height = pheight
        self.color  = pcolor

    def building_graphs(self,pgraph:int,ptitle,pxlabel,px_data,py_data,pylabel,pwidth,pheight,pcolor):
        
        self.__init__(px_data=px_data,py_data=py_data,
                      pwidth=pwidth,pheight=pheight,
                      pcolor=pcolor,pylabel=pylabel,
                      pxlabel=pxlabel,ptitle=ptitle)
            
        if pgraph == Graphs.BARH:
            self.build_barh()
                
    def build_barh(self):
       
        plt.figure(figsize=(self.width,self.height))
        plt.barh(self.x_data,self.y_data, color=self.color)    
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        plt.tight_layout()
    
    def show_graph(self):
        plt.show()