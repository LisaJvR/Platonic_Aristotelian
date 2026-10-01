from sklearn.linear_model import LinearRegression
import numpy as np

data ={
    "Platonic": {
        "mknn_k10": {
            "imagenet21k": {
                "tiny": {
                    "bloom560m": 0.085,
                    "bloom1.1b": 0.090,
                    "bloom1.7b": 0.095,
                    "bloom3b": 0.096,
                    "bloom7b": 0.104,
                    "openllama3b": 0.120,
                    "openllama7b": 0.125,
                    "openllama13b": 0.127,
                    "llama7b": 0.128,
                    "llama13b": 0.127,
                    "llama33b": 0.128,
                    "llama65b": 0.130
                },

                "small": {
                    "bloom560m": 0.092,
                    "bloom1.1b": 0.096,
                    "bloom1.7b": 0.102,
                    "bloom3b": 0.104,
                    "bloom7b": 0.112,
                    "openllama3b": 0.131,
                    "openllama7b": 0.134,
                    "openllama13b": 0.138,
                    "llama7b": 0.140,
                    "llama13b": 0.137,
                    "llama33b": 0.140,
                    "llama65b": 0.142
                },

                "base": {
                    "bloom560m": 0.094,
                    "bloom1.1b": 0.097,
                    "bloom1.7b": 0.102,
                    "bloom3b": 0.105,
                    "bloom7b": 0.113,
                    "openllama3b": 0.128,
                    "openllama7b": 0.134,
                    "openllama13b": 0.136,
                    "llama7b": 0.139,
                    "llama13b": 0.138,
                    "llama33b": 0.140,
                    "llama65b": 0.142
                },

                "large": {
                    "bloom560m": 0.087,
                    "bloom1.1b": 0.092,
                    "bloom1.7b": 0.097,
                    "bloom3b": 0.099,
                    "bloom7b": 0.105,
                    "openllama3b": 0.123,
                    "openllama7b": 0.129,
                    "openllama13b": 0.132,
                    "llama7b": 0.133,
                    "llama13b": 0.130,
                    "llama33b": 0.135,
                    "llama65b": 0.137
                }
            },
             "mae": {
                            "base": {
                                "bloom560m": 0.055,
                                "bloom1.1b": 0.060,
                                "bloom1.7b": 0.061,
                                "bloom3b": 0.065,
                                "bloom7b": 0.067,
                                "openllama3b": 0.075,
                                "openllama7b": 0.079,
                                "openllama13b": 0.077,
                                "llama7b": 0.077,
                                "llama13b": 0.077,
                                "llama33b": 0.080,
                                "llama65b": 0.079
                            },
                    
                            "large": {
                                "bloom560m": 0.066,
                                "bloom1.1b": 0.073,
                                "bloom1.7b": 0.074,
                                "bloom3b": 0.075,
                                "bloom7b": 0.081,
                                "openllama3b": 0.092,
                                "openllama7b": 0.096,
                                "openllama13b": 0.098,
                                "llama7b": 0.096,
                                "llama13b": 0.096,
                                "llama33b": 0.100,
                                "llama65b": 0.100
                            },
                    
                            "huge": {
                                "bloom560m": 0.066,
                                "bloom1.1b": 0.071,
                                "bloom1.7b": 0.075,
                                "bloom3b": 0.076,
                                "bloom7b": 0.081,
                                "openllama3b": 0.092,
                                "openllama7b": 0.097,
                                "openllama13b": 0.098,
                                "llama7b": 0.096,
                                "llama13b": 0.096,
                                "llama33b": 0.099,
                                "llama65b": 0.098
                            }
                        },
                    
            "dinov2": {
                            "small": {
                                "bloom560m": 0.090,
                                "bloom1.1b": 0.100,
                                "bloom1.7b": 0.103,
                                "bloom3b": 0.106,
                                "bloom7b": 0.113,
                                "openllama3b": 0.134,
                                "openllama7b": 0.137,
                                "openllama13b": 0.141,
                                "llama7b": 0.140,
                                "llama13b": 0.138,
                                "llama33b": 0.141,
                                "llama65b": 0.140
                            },
                    
                            "base": {
                                "bloom560m": 0.098,
                                "bloom1.1b": 0.105,
                                "bloom1.7b": 0.109,
                                "bloom3b": 0.110,
                                "bloom7b": 0.120,
                                "openllama3b": 0.139,
                                "openllama7b": 0.144,
                                "openllama13b": 0.150,
                                "llama7b": 0.150,
                                "llama13b": 0.149,
                                "llama33b": 0.151,
                                "llama65b": 0.151
                            },
                    
                            "large": {
                                "bloom560m": 0.104,
                                "bloom1.1b": 0.110,
                                "bloom1.7b": 0.114,
                                "bloom3b": 0.116,
                                "bloom7b": 0.129,
                                "openllama3b": 0.152,
                                "openllama7b": 0.159,
                                "openllama13b": 0.161,
                                "llama7b": 0.158,
                                "llama13b": 0.155,
                                "llama33b": 0.160,
                                "llama65b": 0.163
                            },
                    
                            "giant": {
                                "bloom560m": 0.104,
                                "bloom1.1b": 0.112,
                                "bloom1.7b": 0.116,
                                "bloom3b": 0.119,
                                "bloom7b": 0.131,
                                "openllama3b": 0.155,
                                "openllama7b": 0.162,
                                "openllama13b": 0.162,
                                "llama7b": 0.162,
                                "llama13b": 0.161,
                                "llama33b": 0.164,
                                "llama65b": 0.164
                            }
                        },
                    
            "clip": {
                            "base": {
                                "bloom560m": 0.118,
                                "bloom1.1b": 0.128,
                                "bloom1.7b": 0.131,
                                "bloom3b": 0.134,
                                "bloom7b": 0.148,
                                "openllama3b": 0.173,
                                "openllama7b": 0.179,
                                "openllama13b": 0.181,
                                "llama7b": 0.184,
                                "llama13b": 0.179,
                                "llama33b": 0.180,
                                "llama65b": 0.184
                            },
                    
                            "large": {
                                "bloom560m": 0.125,
                                "bloom1.1b": 0.135,
                                "bloom1.7b": 0.141,
                                "bloom3b": 0.146,
                                "bloom7b": 0.160,
                                "openllama3b": 0.188,
                                "openllama7b": 0.195,
                                "openllama13b": 0.198,
                                "llama7b": 0.198,
                                "llama13b": 0.196,
                                "llama33b": 0.196,
                                "llama65b": 0.201
                            },
                    
                            "huge": {
                                "bloom560m": 0.128,
                                "bloom1.1b": 0.137,
                                "bloom1.7b": 0.141,
                                "bloom3b": 0.149,
                                "bloom7b": 0.161,
                                "openllama3b": 0.190,
                                "openllama7b": 0.196,
                                "openllama13b": 0.200,
                                "llama7b": 0.202,
                                "llama13b": 0.198,
                                "llama33b": 0.202,
                                "llama65b": 0.205
                            }
                        },
                    
            "clip (12K ft)": {
                            "base": {
                                "bloom560m": 0.098,
                                "bloom1.1b": 0.102,
                                "bloom1.7b": 0.111,
                                "bloom3b": 0.112,
                                "bloom7b": 0.121,
                                "openllama3b": 0.139,
                                "openllama7b": 0.143,
                                "openllama13b": 0.145,
                                "llama7b": 0.148,
                                "llama13b": 0.146,
                                "llama33b": 0.150,
                                "llama65b": 0.152
                            },
                    
                            "large": {
                                "bloom560m": 0.102,
                                "bloom1.1b": 0.110,
                                "bloom1.7b": 0.117,
                                "bloom3b": 0.118,
                                "bloom7b": 0.131,
                                "openllama3b": 0.152,
                                "openllama7b": 0.156,
                                "openllama13b": 0.160,
                                "llama7b": 0.162,
                                "llama13b": 0.159,
                                "llama33b": 0.164,
                                "llama65b": 0.166
                            },
                    
                            "huge": {
                                "bloom560m": 0.109,
                                "bloom1.1b": 0.115,
                                "bloom1.7b": 0.123,
                                "bloom3b": 0.126,
                                "bloom7b": 0.136,
                                "openllama3b": 0.160,
                                "openllama7b": 0.168,
                                "openllama13b": 0.170,
                                "llama7b": 0.171,
                                "llama13b": 0.167,
                                "llama33b": 0.171,
                                "llama65b": 0.174
                            }
                        },
        },
        "cka_linear": {
        
                    "imagenet21k": {
                        "tiny": {
                            "bloom560m": 0.305,
                            "bloom1.1b": 0.330,
                            "bloom1.7b": 0.330,
                            "bloom3b": 0.340,
                            "bloom7b": 0.355,
                            "openllama3b": 0.375,
                            "openllama7b": 0.390,
                            "openllama13b": 0.395,
                            "llama7b": 0.395,
                            "llama13b": 0.390,
                            "llama33b": 0.390,
                            "llama65b": 0.390
                        },
        
                        "small": {
                            "bloom560m": 0.310,
                            "bloom1.1b": 0.335,
                            "bloom1.7b": 0.335,
                            "bloom3b": 0.345,
                            "bloom7b": 0.365,
                            "openllama3b": 0.395,
                            "openllama7b": 0.415,
                            "openllama13b": 0.425,
                            "llama7b": 0.420,
                            "llama13b": 0.425,
                            "llama33b": 0.430,
                            "llama65b": 0.435
                        },
        
                        "base": {
                            "bloom560m": 0.315,
                            "bloom1.1b": 0.340,
                            "bloom1.7b": 0.340,
                            "bloom3b": 0.350,
                            "bloom7b": 0.375,
                            "openllama3b": 0.410,
                            "openllama7b": 0.430,
                            "openllama13b": 0.440,
                            "llama7b": 0.440,
                            "llama13b": 0.445,
                            "llama33b": 0.460,
                            "llama65b": 0.480
                        },
        
                        "large": {
                            "bloom560m": 0.310,
                            "bloom1.1b": 0.335,
                            "bloom1.7b": 0.335,
                            "bloom3b": 0.345,
                            "bloom7b": 0.380,
                            "openllama3b": 0.415,
                            "openllama7b": 0.435,
                            "openllama13b": 0.445,
                            "llama7b": 0.445,
                            "llama13b": 0.450,
                            "llama33b": 0.470,
                            "llama65b": 0.495
                        }
                    },
        
                    "mae": {
                        "base": {
                            "bloom560m": 0.227,
                            "bloom1.1b": 0.249,
                            "bloom1.7b": 0.245,
                            "bloom3b": 0.251,
                            "bloom7b": 0.257,
                            "openllama3b": 0.265,
                            "openllama7b": 0.271,
                            "openllama13b": 0.273,
                            "llama7b": 0.269,
                            "llama13b": 0.267,
                            "llama33b": 0.271,
                            "llama65b": 0.263
                        },
        
                        "large": {
                            "bloom560m": 0.268,
                            "bloom1.1b": 0.297,
                            "bloom1.7b": 0.291,
                            "bloom3b": 0.304,
                            "bloom7b": 0.312,
                            "openllama3b": 0.326,
                            "openllama7b": 0.336,
                            "openllama13b": 0.336,
                            "llama7b": 0.330,
                            "llama13b": 0.331,
                            "llama33b": 0.333,
                            "llama65b": 0.325
                        },
        
                        "huge": {
                            "bloom560m": 0.271,
                            "bloom1.1b": 0.293,
                            "bloom1.7b": 0.289,
                            "bloom3b": 0.301,
                            "bloom7b": 0.309,
                            "openllama3b": 0.328,
                            "openllama7b": 0.339,
                            "openllama13b": 0.339,
                            "llama7b": 0.331,
                            "llama13b": 0.330,
                            "llama33b": 0.333,
                            "llama65b": 0.327
                        }
                    },
        
                    "dinov2": {
                        "small": {
                            "bloom560m": 0.315,
                            "bloom1.1b": 0.340,
                            "bloom1.7b": 0.335,
                            "bloom3b": 0.349,
                            "bloom7b": 0.360,
                            "openllama3b": 0.378,
                            "openllama7b": 0.38,
                            "openllama13b": 0.38,
                            "llama7b": 0.378,
                            "llama13b": 0.381,
                            "llama33b": 0.39,
                            "llama65b": 0.398
                        },
        
                        "base": {
                            "bloom560m": 0.330,
                            "bloom1.1b": 0.360,
                            "bloom1.7b": 0.355,
                            "bloom3b": 0.370,
                            "bloom7b": 0.385,
                            "openllama3b": 0.405,
                            "openllama7b": 0.415,
                            "openllama13b": 0.415,
                            "llama7b": 0.405,
                            "llama13b": 0.410,
                            "llama33b": 0.425,
                            "llama65b": 0.445
                        },
        
                        "large": {
                            "bloom560m": 0.335,
                            "bloom1.1b": 0.360,
                            "bloom1.7b": 0.355,
                            "bloom3b": 0.375,
                            "bloom7b": 0.395,
                            "openllama3b": 0.425,
                            "openllama7b": 0.440,
                            "openllama13b": 0.450,
                            "llama7b": 0.445,
                            "llama13b": 0.445,
                            "llama33b": 0.455,
                            "llama65b": 0.465
                        },
        
                        "giant": {
                            "bloom560m": 0.335,
                            "bloom1.1b": 0.360,
                            "bloom1.7b": 0.355,
                            "bloom3b": 0.375,
                            "bloom7b": 0.395,
                            "openllama3b": 0.425,
                            "openllama7b": 0.445,
                            "openllama13b": 0.455,
                            "llama7b": 0.450,
                            "llama13b": 0.455,
                            "llama33b": 0.465,
                            "llama65b": 0.475
                        }
                    },
        
                    "clip": {
                        "base": {
                            "bloom560m": 0.380,
                            "bloom1.1b": 0.420,
                            "bloom1.7b": 0.410,
                            "bloom3b": 0.430,
                            "bloom7b": 0.445,
                            "openllama3b": 0.480,
                            "openllama7b": 0.495,
                            "openllama13b": 0.500,
                            "llama7b": 0.500,
                            "llama13b": 0.495,
                            "llama33b": 0.495,
                            "llama65b": 0.495
                        },
        
                        "large": {
                            "bloom560m": 0.390,
                            "bloom1.1b": 0.425,
                            "bloom1.7b": 0.425,
                            "bloom3b": 0.445,
                            "bloom7b": 0.465,
                            "openllama3b": 0.495,
                            "openllama7b": 0.515,
                            "openllama13b": 0.520,
                            "llama7b": 0.515,
                            "llama13b": 0.515,
                            "llama33b": 0.515,
                            "llama65b": 0.515
                        },
        
                        "huge": {
                            "bloom560m": 0.400,
                            "bloom1.1b": 0.430,
                            "bloom1.7b": 0.430,
                            "bloom3b": 0.450,
                            "bloom7b": 0.470,
                            "openllama3b": 0.505,
                            "openllama7b": 0.525,
                            "openllama13b": 0.530,
                            "llama7b": 0.525,
                            "llama13b": 0.525,
                            "llama33b": 0.525,
                            "llama65b": 0.530
                        }
                    },
        
                    "clip (12K ft)": {
                        "base": {
                            "bloom560m": 0.325,
                            "bloom1.1b": 0.345,
                            "bloom1.7b": 0.355,
                            "bloom3b": 0.355,
                            "bloom7b": 0.375,
                            "openllama3b": 0.410,
                            "openllama7b": 0.430,
                            "openllama13b": 0.445,
                            "llama7b": 0.450,
                            "llama13b": 0.445,
                            "llama33b": 0.460,
                            "llama65b": 0.485
                        },
        
                        "large": {
                            "bloom560m": 0.335,
                            "bloom1.1b": 0.360,
                            "bloom1.7b": 0.360,
                            "bloom3b": 0.375,
                            "bloom7b": 0.400,
                            "openllama3b": 0.435,
                            "openllama7b": 0.460,
                            "openllama13b": 0.470,
                            "llama7b": 0.470,
                            "llama13b": 0.465,
                            "llama33b": 0.480,
                            "llama65b": 0.505
                        },
        
                        "huge": {
                            "bloom560m": 0.345,
                            "bloom1.1b": 0.370,
                            "bloom1.7b": 0.370,
                            "bloom3b": 0.385,
                            "bloom7b": 0.410,
                            "openllama3b": 0.450,
                            "openllama7b": 0.475,
                            "openllama13b": 0.485,
                            "llama7b": 0.490,
                            "llama13b": 0.485,
                            "llama33b": 0.500,
                            "llama65b": 0.525
                        }
                    }
                },
"svcca": {

    "imagenet21k": {

        "tiny": {
            "bloom560m": 0.343,
            "bloom1.1b": 0.379,
            "bloom1.7b": 0.376,
            "bloom3b": 0.378,
            "bloom7b": 0.373,
            "openllama3b": 0.406,
            "openllama7b": 0.405,
            "openllama13b": 0.403,
            "llama7b": 0.409,
            "llama13b": 0.406,
            "llama33b": 0.409,
            "llama65b": 0.409
        },

        "small": {
            "bloom560m": 0.349,
            "bloom1.1b": 0.359,
            "bloom1.7b": 0.371,
            "bloom3b": 0.383,
            "bloom7b": 0.377,
            "openllama3b": 0.407,
            "openllama7b": 0.389,
            "openllama13b": 0.398,
            "llama7b": 0.410,
            "llama13b": 0.405,
            "llama33b": 0.431,
            "llama65b": 0.440
        },

        "base": {
            "bloom560m": 0.337,
            "bloom1.1b": 0.345,
            "bloom1.7b": 0.338,
            "bloom3b": 0.352,
            "bloom7b": 0.377,
            "openllama3b": 0.368,
            "openllama7b": 0.381,
            "openllama13b": 0.392,
            "llama7b": 0.405,
            "llama13b": 0.402,
            "llama33b": 0.397,
            "llama65b": 0.415
        },

        "large": {
            "bloom560m": 0.334,
            "bloom1.1b": 0.340,
            "bloom1.7b": 0.354,
            "bloom3b": 0.357,
            "bloom7b": 0.365,
            "openllama3b": 0.379,
            "openllama7b": 0.377,
            "openllama13b": 0.395,
            "llama7b": 0.397,
            "llama13b": 0.396,
            "llama33b": 0.399,
            "llama65b": 0.409
        }
    },

    "mae": {

        "base": {
            "bloom560m": 0.317,
            "bloom1.1b": 0.315,
            "bloom1.7b": 0.315,
            "bloom3b": 0.316,
            "bloom7b": 0.323,
            "openllama3b": 0.335,
            "openllama7b": 0.321,
            "openllama13b": 0.354,
            "llama7b": 0.345,
            "llama13b": 0.340,
            "llama33b": 0.345,
            "llama65b": 0.360
        },

        "large": {
            "bloom560m": 0.348,
            "bloom1.1b": 0.355,
            "bloom1.7b": 0.355,
            "bloom3b": 0.363,
            "bloom7b": 0.358,
            "openllama3b": 0.379,
            "openllama7b": 0.374,
            "openllama13b": 0.386,
            "llama7b": 0.382,
            "llama13b": 0.388,
            "llama33b": 0.391,
            "llama65b": 0.391
        },

        "huge": {
            "bloom560m": 0.351,
            "bloom1.1b": 0.359,
            "bloom1.7b": 0.371,
            "bloom3b": 0.364,
            "bloom7b": 0.387,
            "openllama3b": 0.389,
            "openllama7b": 0.406,
            "openllama13b": 0.389,
            "llama7b": 0.400,
            "llama13b": 0.397,
            "llama33b": 0.395,
            "llama65b": 0.393
        }
    },

    "dinov2": {

        "small": {
            "bloom560m": 0.389,
            "bloom1.1b": 0.400,
            "bloom1.7b": 0.390,
            "bloom3b": 0.410,
            "bloom7b": 0.415,
            "openllama3b": 0.430,
            "openllama7b": 0.429,
            "openllama13b": 0.432,
            "llama7b": 0.429,
            "llama13b": 0.430,
            "llama33b": 0.454,
            "llama65b": 0.447
        },

        "base": {
            "bloom560m": 0.410,
            "bloom1.1b": 0.423,
            "bloom1.7b": 0.421,
            "bloom3b": 0.423,
            "bloom7b": 0.431,
            "openllama3b": 0.463,
            "openllama7b": 0.434,
            "openllama13b": 0.448,
            "llama7b": 0.452,
            "llama13b": 0.445,
            "llama33b": 0.463,
            "llama65b": 0.456
        },

        "large": {
            "bloom560m": 0.404,
            "bloom1.1b": 0.412,
            "bloom1.7b": 0.406,
            "bloom3b": 0.422,
            "bloom7b": 0.431,
            "openllama3b": 0.453,
            "openllama7b": 0.438,
            "openllama13b": 0.462,
            "llama7b": 0.464,
            "llama13b": 0.445,
            "llama33b": 0.461,
            "llama65b": 0.463
        },

        "giant": {
            "bloom560m": 0.378,
            "bloom1.1b": 0.399,
            "bloom1.7b": 0.413,
            "bloom3b": 0.416,
            "bloom7b": 0.416,
            "openllama3b": 0.438,
            "openllama7b": 0.436,
            "openllama13b": 0.441,
            "llama7b": 0.442,
            "llama13b": 0.446,
            "llama33b": 0.466,
            "llama65b": 0.455
        }
    },

    "clip": {

        "base": {
            "bloom560m": 0.424,
            "bloom1.1b": 0.488,
            "bloom1.7b": 0.468,
            "bloom3b": 0.472,
            "bloom7b": 0.472,
            "openllama3b": 0.510,
            "openllama7b": 0.495,
            "openllama13b": 0.510,
            "llama7b": 0.513,
            "llama13b": 0.484,
            "llama33b": 0.493,
            "llama65b": 0.480
        },

        "large": {
            "bloom560m": 0.445,
            "bloom1.1b": 0.455,
            "bloom1.7b": 0.475,
            "bloom3b": 0.477,
            "bloom7b": 0.491,
            "openllama3b": 0.503,
            "openllama7b": 0.512,
            "openllama13b": 0.514,
            "llama7b": 0.514,
            "llama13b": 0.527,
            "llama33b": 0.523,
            "llama65b": 0.538
        },

        "huge": {
            "bloom560m": 0.449,
            "bloom1.1b": 0.470,
            "bloom1.7b": 0.468,
            "bloom3b": 0.516,
            "bloom7b": 0.499,
            "openllama3b": 0.534,
            "openllama7b": 0.519,
            "openllama13b": 0.513,
            "llama7b": 0.513,
            "llama13b": 0.514,
            "llama33b": 0.525,
            "llama65b": 0.528
        }
    },

    "clip (12K ft)": {

        "base": {
            "bloom560m": 0.363,
            "bloom1.1b": 0.390,
            "bloom1.7b": 0.385,
            "bloom3b": 0.374,
            "bloom7b": 0.384,
            "openllama3b": 0.415,
            "openllama7b": 0.423,
            "openllama13b": 0.420,
            "llama7b": 0.443,
            "llama13b": 0.431,
            "llama33b": 0.440,
            "llama65b": 0.437
        },

        "large": {
            "bloom560m": 0.366,
            "bloom1.1b": 0.378,
            "bloom1.7b": 0.410,
            "bloom3b": 0.397,
            "bloom7b": 0.394,
            "openllama3b": 0.425,
            "openllama7b": 0.421,
            "openllama13b": 0.418,
            "llama7b": 0.449,
            "llama13b": 0.450,
            "llama33b": 0.444,
            "llama65b": 0.439
        },

        "huge": {
            "bloom560m": 0.371,
            "bloom1.1b": 0.406,
            "bloom1.7b": 0.409,
            "bloom3b": 0.424,
            "bloom7b": 0.429,
            "openllama3b": 0.450,
            "openllama7b": 0.435,
            "openllama13b": 0.451,
            "llama7b": 0.447,
            "llama13b": 0.455,
            "llama33b": 0.469,
            "llama65b": 0.470
        }
    }
}
    }
}

def get_platonic_trend(x, type= "Platonic", metric="mknn_k10"):
    print(f"Calculating trend for {type} | {metric}...")
    all_coeffs = {}
    avg_family_coeffs = {}

    for family in data[type][metric]:
        for image_model in data[type][metric][family]:
            
            y = np.array(list(data[type][metric][family][image_model].values()))
            # print(
            #     f"{family} | {image_model}: "
            #     f"x={len(x)}, y={len(y)}"
            # )

            if len(x) != len(y):
                print(
                    f"Warning: Length mismatch for {family} | {image_model}: "
                    f"x has length {len(x)}, y has length {len(y)}. "
                    "Truncating to the shorter length."
                )
                # continue
            if len(x) < len(y):
                y = y[:len(x)]
            else:
                x = x[:len(y)]

            coeff = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1)).coef_[0]

            all_coeffs[(family, image_model)] = coeff[0]

        avg_family_coeff = np.mean([all_coeffs[(family, image_model)] for image_model in data[type][metric][family]])
        avg_family_coeffs[family] = avg_family_coeff

        if metric == "svcca":
            print(f"Average trend for {family}: {avg_family_coeff:.4f}")
    return all_coeffs, avg_family_coeffs
