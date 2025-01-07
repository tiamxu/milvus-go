package milvus

type Config struct {
	MilvusAddr string `yaml:"milvus_addr"`
	UserName   string `yaml:"username"`
	Password   string `yaml:"password"`
}
