package milvus

import (
	"context"
	"fmt"
	"log"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

type MilvusClient struct {
	client client.Client
}

func NewMilvusClient(address string) (*MilvusClient, error) {
	ctx := context.Background()
	c, err := client.NewClient(ctx, client.Config{
		Address: address,
	})
	if err != nil {
		return nil, fmt.Errorf("连接milvus服务器失败: %w", err)
	}
	return &MilvusClient{client: c}, nil
}

func (mc *MilvusClient) Close() {
	mc.client.Close()
}

func (mc *MilvusClient) CreateCollection(ctx context.Context, name string, schema *entity.Schema) error {
	ok, err := mc.client.HasCollection(ctx, name)
	if err != nil {
		return fmt.Errorf("无法检查集合是否存在： %w", err)
	}
	if ok {
		log.Printf("集合 %s 已存在\n", name)
		_ = mc.client.DropCollection(ctx, name)
		return nil
	}

	if err := mc.client.CreateCollection(ctx, schema, entity.DefaultShardNumber); err != nil {
		return fmt.Errorf("创建集合失败: %w", err)
	}
	log.Printf("集合 %s 创建成功\n", name)

	return nil
}

func (mc *MilvusClient) Insert(ctx context.Context, collectionName string, partitionName string, columns ...entity.Column) error {
	_, err := mc.client.Insert(ctx, collectionName, partitionName, columns...)
	if err != nil {
		return fmt.Errorf("插入数据失败: %w", err)
	}
	log.Printf("inserted entities into collection `%s`\n", collectionName)
	return nil
}

func (mc *MilvusClient) Flush(ctx context.Context, collectionName string) error {
	if err := mc.client.Flush(ctx, collectionName, false); err != nil {
		return fmt.Errorf("flush data failed: %w", err)
	}
	log.Printf("flushed data of collection `%s`\n", collectionName)
	return nil
}

func (mc *MilvusClient) CreateIndex(ctx context.Context, collectionName, fieldName string, idx entity.Index) error {
	// idx, err := entity.NewIndexIvfFlat(entity.L2, 128)
	// if err != nil {
	// 	return fmt.Errorf("failed to create ivf flat index: %w", err)
	// }
	if err := mc.client.CreateIndex(ctx, collectionName, fieldName, idx, false); err != nil {
		return fmt.Errorf("create index failed: %w", err)
	}
	log.Printf("create index on field %s of collection %s\n", fieldName, collectionName)
	return nil
}

func (mc *MilvusClient) LoadCollection(ctx context.Context, collectionName string) error {
	err := mc.client.LoadCollection(ctx, collectionName, false)
	if err != nil {
		return fmt.Errorf("load collection failed: %w", err)
	}
	log.Printf("loaded collection `%s`\n", collectionName)
	return nil
}
func (mc *MilvusClient) Delete(ctx context.Context, collectionName string, pks *entity.ColumnInt64) error {
	if err := mc.client.DeleteByPks(ctx, collectionName, "", pks); err != nil {
		return fmt.Errorf("delete entities failed: %w", err)
	}
	return nil
}

func (mc *MilvusClient) Drop(ctx context.Context, name string) error {
	if err := mc.client.DropCollection(ctx, name); err != nil {
		return fmt.Errorf("drop collection failed: %w", err)
	}
	log.Printf("集合 %s 删除成功\n", name)

	return nil
}

func GenerateColumnData(colName string, colType entity.FieldType, values interface{}) (entity.Column, error) {
	switch colType {
	case entity.FieldTypeInt64:
		v, ok := values.([]int64)
		if !ok {
			return nil, fmt.Errorf("type assertion to []int64 failed")
		}
		return entity.NewColumnInt64(colName, v), nil
	case entity.FieldTypeDouble:
		v, ok := values.([]float64)
		if !ok {
			return nil, fmt.Errorf("type assertion to []float64 failed")
		}
		return entity.NewColumnDouble(colName, v), nil
	case entity.FieldTypeFloatVector:
		v, ok := values.([][]float32)
		if !ok {
			return nil, fmt.Errorf("type assertion to [][]float32 failed")
		}
		dim := len(v[0])
		for _, vec := range v {
			if len(vec) != dim {
				return nil, fmt.Errorf("inconsistent vector dimensions")
			}
		}
		return entity.NewColumnFloatVector(colName, dim, v), nil
	default:
		return nil, fmt.Errorf("unsupported column type: %v", colType)
	}
}
