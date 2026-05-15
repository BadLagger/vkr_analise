import json

class DataPreparator:
    def __init__(self):
        self.__elements__ = []
    
    def add(self, source_name, dest_name, prep_type, value):
        if len(self.__elements__) != 0:
            for el in self.__elements__:
                if el["src"] == source_name:
                    raise ValueError(f"Dublicated source parameter: {source_name}")
                if el["dest"] == dest_name:
                    raise ValueError(f"Dublicated destination parameter: {dest_name}")
        
        if prep_type not in ("multiplier", "divisor"):
            raise ValueError(f"Unsupported preparation type: {prep_type}")
        
        self.__elements__.append({
            "src": source_name,
            "dest": dest_name,
            "type": prep_type,
            "value": value
        })
    
    def save_to_file(self, f_path):
        try:
            with open(f_path, "w", encoding='utf-8') as o_file:
                json.dump({"parameters":self.__elements__}, o_file, indent=4, ensure_ascii=False)
        except Exception as expt:
            raise Exception(expt)